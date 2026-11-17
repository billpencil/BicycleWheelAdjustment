import json, numpy as np, torch, onnx, onnxruntime as ort

from main_v1 import WheelTruingActorCritic, WheelTruingEnv, Config

N = 32                                   # spokes in the trained wheel

# 32-spoke rear disc wheel example specifications
specs = np.array([
    602.0,  # ERD
    142.0,  # Axle width
    32.5,   # Left Locknut->flange
    35.0,   # Left Flange->center
    19.0,   # Right Flange->center
    48.0,   # Right Locknut->flange
    58.0,   # Left flange diam
    58.0,   # Right flange diam
    2.5,    # Spoke hole diam
    32.0,   # Number of spokes
    3.0,    # Lacing pattern (3-cross)
    2.0,    # Spoke diameter
    0.0     # 0.0=round, 1.0=flattened
])

# Goal tolerances
goals = np.array([
    0.1,    # Peak-to-peak Lateral Runout <= 0.1 mm
    0.1,    # Peak-to-peak Radial Runout <= 0.1 mm
    900.0,  # Left Spoke Min Tension (N)
    1200.0, # Left Spoke Max Tension (N)
    700.0,  # Right Spoke Min Tension (N)
    1000.0  # Right Spoke Max Tension (N)
])
cfg = Config()
env = WheelTruingEnv(specs, goals, gamma=cfg.gamma, seed=10_000 + cfg.seed)

model = WheelTruingActorCritic(in_ch=env.N_CHANNELS)      # same constructor args as in training
model.load_state_dict(torch.load("best_truing_agent.pt", map_location="cpu"))
model.eval()

class Export(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, obs):
        out = self.m(obs)
        return out[0], out[1].reshape(-1)    # logits [B,2N], value [B]

wrapped = Export(model).eval()
dummy = torch.zeros(1, 5, N)
print("logits shape:", wrapped(dummy)[0].shape)   # expect (1, 2N)

torch.onnx.export(
    wrapped, dummy, "wheel_truing.onnx",
    input_names=["obs"], output_names=["logits", "value"],
    dynamic_axes={"obs": {0: "batch"}, "logits": {0: "batch"}, "value": {0: "batch"}},
    opset_version=17,
)

# verify ONNX against PyTorch
sess = ort.InferenceSession("wheel_truing.onnx", providers=["CPUExecutionProvider"])
x = np.random.randn(4, 5, N).astype(np.float32)
with torch.no_grad():
    ref = wrapped(torch.from_numpy(x))[0].numpy()
onnx_out = sess.run(None, {"obs": x})[0]
print("max abs diff:", np.abs(ref - onnx_out).max())    # should be < 1e-4

# constants the GUI needs (copy from your env instance)
cfg = dict(
    n=N,
    goals=[float(g) for g in env.goals[:2]],       # mm
    t_mid=np.asarray(env.t_mid).tolist(),          # N (scalar or per spoke)
    t_half=np.asarray(env.t_half).tolist(),
    max_steps=int(env.max_steps),
    side=np.asarray(env.side).tolist(),
    dT=20.0,                                       # N per quarter-turn, nominal; replace after calibration
)
json.dump(cfg, open("config.json", "w"), indent=1)

m = onnx.load("wheel_truing.onnx")        # automatically picks up wheel_truing.onnx.data
onnx.save_model(m, "wheel_truing_single.onnx", save_as_external_data=False)

ort.InferenceSession("wheel_truing_single.onnx", providers=["CPUExecutionProvider"])
print("ok")