import sys, torch, torch.nn.functional as F
from aiter.ops.opus_gdn_prefill import opus_gdn_prefill_fwd
from aiter.ops.triton.gated_delta_net.gated_delta_rule import chunk_gated_delta_rule
B,T,H = int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4])
which=sys.argv[1]; D=128;dev="cuda";torch.manual_seed(0)
q=F.normalize(torch.randn(B,T,H,D,dtype=torch.bfloat16,device=dev),p=2,dim=-1)
k=F.normalize(torch.randn(B,T,H,D,dtype=torch.bfloat16,device=dev),p=2,dim=-1)
v=torch.randn(B,T,H,D,dtype=torch.bfloat16,device=dev)*0.5
g=F.logsigmoid(torch.randn(B,T,H,dtype=torch.float32,device=dev))
beta=torch.rand(B,T,H,dtype=torch.float32,device=dev)
fn=(lambda: opus_gdn_prefill_fwd(q,k,v,g,beta,BT=64,k1_algo=1,k2_mode=0)) if which=="opus" else (lambda: chunk_gated_delta_rule(q,k,v,g,beta))
for _ in range(5): fn()
torch.cuda.synchronize()
for _ in range(20): fn()
torch.cuda.synchronize()
