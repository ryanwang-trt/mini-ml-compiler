import torch
from torch import nn

torch.manual_seed(7)

model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 2)
)

model.eval()

x = torch.tensor([[0.2, -0.4, 0.7, 1.0]])

with torch.no_grad():
    y = model(x)

with open("model_weights.h", "w") as f:
    f.write("#pragma once\n\n")

    for i, layer in enumerate(model):
        if not isinstance(layer, nn.Linear):
            continue

        w = layer.weight.detach().numpy()
        b = layer.bias.detach().numpy()

        f.write(f"const float w{i}[{w.shape[0]}][{w.shape[1]}] = {{\n")

        for row in w:
            nums = ", ".join(f"{float(v):.9g}f" for v in row)
            f.write(f"    {{{nums}}},\n")

        f.write("};\n\n")

        nums = ", ".join(f"{float(v):.9g}f" for v in b)
        f.write(f"const float b{i}[{len(b)}] = {{{nums}}};\n\n")

    nums = ", ".join(f"{float(v):.9g}f" for v in x[0])
    f.write(f"const float test_input[4] = {{{nums}}};\n")

    nums = ", ".join(f"{float(v):.9g}f" for v in y[0])
    f.write(f"const float expected[2] = {{{nums}}};\n")

print("Output:", y)
print("Exported model_weights.h")
