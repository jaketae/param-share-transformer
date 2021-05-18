from torch import nn


class ParameterSharedTransformerEncoder(nn.TransformerEncoder):
    def __init__(
        self,
        d_model=512,
        nhead=16,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        num_unique_layers=3,
        num_total_layers=6,
        mode="cycle_rev",
        norm=False,
    ):
        assert mode in {"sequence", "cycle", "cycle_rev"}
        quotient, remainder = divmod(num_total_layers, num_unique_layers)
        assert remainder == 0
        if mode == "cycle_rev":
            assert quotient == 2
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation
        )
        super().__init__(encoder_layer, num_layers=num_unique_layers, norm=norm)
        self.N = num_total_layers
        self.M = num_unique_layers
        self.mode = mode
        self.norm = nn.LayerNorm(d_model) if norm else None

    def forward(self, x, mask=None, src_key_padding_mask=None, verbose=False):
        for i in range(self.N):
            if self.mode == "sequence":
                i = i // (self.N // self.M)
            elif self.mode == "cycle":
                i = i % self.M
            elif i > (self.N - 1) / 2:
                i = self.N - i - 1
            if verbose:
                print(f"layer {i}")
            x = self.layers[i](x, mask, src_key_padding_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x
