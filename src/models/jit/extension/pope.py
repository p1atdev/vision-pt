import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_pope(
    inputs: torch.Tensor,  # (batch_size, num_heads, seq_len, dim)
    freqs_cis: torch.Tensor,  # (batch_size, seq_len, dim) complex64
    learned_bias: torch.Tensor | None = None,  # (num_heads, dim,) float32
) -> torch.Tensor:
    _batch_size, num_heads, _seq_len, dim = inputs.shape

    with torch.autocast(device_type="cuda", enabled=False):
        # 1. softplus
        # 2. insert zero imaginary part
        inputs_cis = F.softplus(inputs.float()).to(
            torch.complex64
        )  # (batch_size, num_heads, seq_len, dim) complex64

        # 3. apply learned bias if provided
        freqs_cis = freqs_cis.unsqueeze(1)  # (batch_size, 1, seq_len, dim)
        if learned_bias is not None:
            bias_cis = (
                torch.polar(
                    abs=torch.ones_like(learned_bias).float(),
                    angle=learned_bias.float(),
                )
                .to(torch.complex64)
                .view(1, num_heads, 1, dim)
            )
            # [batch_size(1), num_heads, seq_len(1), dim] complex64

            freqs_cis = freqs_cis * bias_cis  # rotate frequencies by learned bias

        # 4. apply pope frequencies
        output = torch.view_as_real(inputs_cis * freqs_cis).flatten(3)

        return output.type_as(inputs)  # (batch_size, num_heads, seq_len, dim*2) float32


class PopeEmbedder:
    def __init__(
        self,
        pope_theta: float = 256.0,  # ref: Z-Image
        axes_dims: list[int] = [64, 128, 128],  # text, height, width
        axes_lens: list[int] = [256, 128, 128],  # text, height, width
        zero_centered: list[bool] = [False, True, True],  # text, height, width
    ):
        self.pope_theta = pope_theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        self.zero_centered = zero_centered
        self.num_axes = len(axes_dims)

        # text starts with 0, image axes are zero-centered

        self.freqs_cis = self.precompute_freqs_cis(
            theta=self.pope_theta,
            dims=self.axes_dims,
            lens=self.axes_lens,
        )

    def get_offset(self, axis: int) -> int:
        """Get position offset for given axis."""
        return self.axes_lens[axis] // 2 if self.zero_centered[axis] else 0

    @staticmethod
    def get_pope_freqs(
        dim: int,
        min_position: int = 0,
        max_position: int = 128,
        theta: float = 10000.0,
    ) -> torch.Tensor:
        freqs = 1.0 / (
            theta
            ** (
                # pope uses full dim
                torch.arange(0, dim, 1, dtype=torch.float64, device=torch.device("cpu"))
                / dim
            )
        )
        positions = torch.arange(
            start=min_position,
            end=max_position,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )

        freqs = torch.outer(positions, freqs).float()  # (max_position, dim)

        freqs_cis = torch.polar(
            abs=torch.ones_like(freqs),
            angle=freqs,
        ).to(torch.complex64)  # (min_position~max_position, dim//2) complex64

        # 大きさは変えずに回転を表す複素数
        return freqs_cis

    @staticmethod
    def precompute_freqs_cis(
        theta: float,
        dims: list[int],
        lens: list[int],
        zero_centered: list[bool] = [False, True, True],
    ):
        freqs_cis = []

        for i, (dim, len_) in enumerate(zip(dims, lens)):
            freq_cis = PopeEmbedder.get_pope_freqs(
                dim=dim,
                # len_: 128 -> -64 to 63
                min_position=(len_ // 2) - len_ if zero_centered[i] else 0,
                max_position=(len_ // 2) if zero_centered[i] else len_,
                theta=theta,
            )  # (len_, dim) complex64

            freqs_cis.append(freq_cis)

        return freqs_cis

    # get frequencies for given position ids
    def __call__(self, position_ids: torch.Tensor):
        # move to device
        freqs_cis = [fc.to(position_ids.device) for fc in self.freqs_cis]

        result = []
        for i in range(len(self.axes_dims)):
            index = (
                position_ids[..., i : i + 1]
                .repeat(
                    # match dimensions for each axis
                    1,  # batch size?
                    1,  # sequence length?
                    freqs_cis[i].shape[-1],
                )
                .to(torch.int64)
            ) + self.get_offset(axis=i)  # adjust for zero-centered axes
            result.append(
                torch.gather(
                    freqs_cis[i].unsqueeze(0),
                    dim=1,
                    index=index,
                )
            )

        return torch.cat(result, dim=-1)

    def prepare_image_position_ids(
        self,
        height: int,
        width: int,
        patch_size,
        global_index: int,
    ) -> torch.Tensor:
        # [H/patch_size, W/patch_size]

        h_patches = height // patch_size
        w_patches = width // patch_size

        position_ids = torch.zeros(
            h_patches,
            w_patches,
            self.num_axes,
        )

        # image_index
        position_ids[:, :, 0] = global_index  # image

        # height (y-index)
        position_ids[:, :, 1] = (
            torch.arange(
                start=h_patches // 2 - h_patches,
                end=h_patches // 2,
            )
            .unsqueeze(1)
            .repeat(1, w_patches)
        )
        # width (x-index)
        position_ids[:, :, 2] = (
            torch.arange(
                start=w_patches // 2 - w_patches,
                end=w_patches // 2,
            )
            .unsqueeze(0)
            .repeat(h_patches, 1)
        )

        return position_ids.view(-1, self.num_axes)  # (num_patches, n_axes)

    def prepare_context_position_ids(
        self,
        seq_len: int,
        global_index: int = 0,
    ) -> torch.Tensor:
        position_ids = torch.zeros(
            seq_len,
            self.num_axes,
        )

        # context_index (i, ..., i)
        position_ids[:, 0] = global_index  # text

        # token indices are (0, 0)...(seq_len-1, seq_len-1)
        position_ids[:, 1] = torch.arange(seq_len)
        position_ids[:, 2] = torch.arange(seq_len)

        return position_ids


class NormalizedPopeEmbedder(PopeEmbedder):
    def __init__(
        self,
        pope_theta: float = 256.0,  # ref: Z-Image
        axes_dims: list[int] = [64, 128, 128],  # text, height, width
        axes_lens: list[int] = [256, 128, 128],  # text, height, width
        zero_centered: list[bool] = [False, True, True],  # text, height, width
        do_normalize: list[bool] = [
            False,  # text axis disabled
            True,
            True,  # normalized by 64 tokens for height and width
        ],
        normalize_by: float = 64.0,
    ):
        self.pope_theta = pope_theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        self.zero_centered = zero_centered
        self.do_normalize = do_normalize
        self.normalize_by = normalize_by
        self.num_axes = len(axes_dims)

        self.freqs_cis = self.precompute_freqs_cis(
            theta=self.pope_theta,
            dims=self.axes_dims,
            lens=self.axes_lens,
        )  # but only used with text axis

    def get_normalized_pope_freqs(
        self,
        dim: int,
        positions: torch.Tensor,  # (num_positions,)
    ) -> torch.Tensor:
        freqs = 1.0 / (
            self.pope_theta
            ** (
                # pope uses full dim
                torch.arange(0, dim, 1, dtype=torch.float64, device=torch.device("cpu"))
                / dim
            )
        )

        # normalize positions
        if (positions.max() - positions.min()) != 0:
            positions = (
                positions / (positions.max() - positions.min()) * self.normalize_by
            )  # normalize positions

        freqs = torch.outer(positions, freqs).float()  # (max_position, dim)

        freqs_cis = torch.polar(
            abs=torch.ones_like(freqs),
            angle=freqs,
        ).to(torch.complex64)  # (min_position~max_position, dim//2) complex64

        # 大きさは変えずに回転を表す複素数
        return freqs_cis  # (num_positions, dim) complex64

    # get frequencies for given position ids
    def __call__(self, position_ids: torch.Tensor):
        # move to device
        freqs_cis = [fc.to(position_ids.device) for fc in self.freqs_cis]

        result = []
        for i, do_norm in enumerate(self.do_normalize):
            if not do_norm:
                # not normalized
                # use default pope frequencies
                index = (
                    position_ids[..., i : i + 1]
                    .repeat(
                        # match dimensions for each axis
                        1,  # batch size?
                        1,  # sequence length?
                        freqs_cis[i].shape[-1],
                    )
                    .to(torch.int64)
                ) + self.get_offset(axis=i)  # adjust for zero-centered axes
                result.append(
                    torch.gather(
                        freqs_cis[i].unsqueeze(0),
                        dim=1,
                        index=index,
                    )
                )
                continue

            # normalized pope frequencies
            # example position ids:
            # [-2, -1, 0, 1, 2] (for 5 tokens)
            # [-1.5, -0.5, 0.5, 1.5] (for 4 tokens)
            result.append(
                # normalized to
                # [-2/4*64, -1/4*64, 0/4*64, 1/4*64, 2/4*64] = [-32, -16, 0, 16, 32]
                # [-1.5/3*64, -0.5/3*64, 0.5/3*64, 1.5/3*64] = [-32, -10.67, 10.67, 32]
                self.get_normalized_pope_freqs(
                    dim=self.axes_dims[i],
                    positions=position_ids[..., i].float(),
                ).unsqueeze(0)
            )

        return torch.cat(result, dim=-1)

    # normalized image position
    def prepare_image_position_ids(
        self,
        height: int,
        width: int,
        patch_size,
        global_index: int,
    ) -> torch.Tensor:
        # [H/patch_size, W/patch_size]

        h_patches = height // patch_size
        w_patches = width // patch_size

        position_ids = torch.zeros(
            h_patches,
            w_patches,
            self.num_axes,
        )

        # image_index
        position_ids[:, :, 0] = global_index  # image

        # height (y-index)
        position_ids[:, :, 1] = (
            (torch.arange(h_patches, dtype=torch.float32) - ((h_patches - 1) / 2))
            .unsqueeze(1)
            .repeat(1, w_patches)
        )
        # width (x-index)
        position_ids[:, :, 2] = (
            (torch.arange(w_patches, dtype=torch.float32) - ((w_patches - 1) / 2))
            .unsqueeze(0)
            .repeat(h_patches, 1)
        )

        return position_ids.view(-1, self.num_axes)  # (num_patches, n_axes)

    # same as parent
    def prepare_context_position_ids(
        self,
        seq_len: int,
        global_index: int = 0,
    ) -> torch.Tensor:
        position_ids = torch.zeros(
            seq_len,
            self.num_axes,
        )

        # context_index (i, ..., i)
        position_ids[:, 0] = global_index  # text

        # token indices are (0, 0)...(seq_len-1, seq_len-1)
        position_ids[:, 1] = torch.arange(seq_len)
        position_ids[:, 2] = torch.arange(seq_len)

        return position_ids
