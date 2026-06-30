import sqlite3, glob, sys, re
NITERS=25
OPUS_K14=["gdn_k1"]
OPUS_K56=["chunk_gated_delta_rule_fwd_h","gdn_k2"]
TRI_K14=["cumsum","prefix_scan","kkt","inverse","recompute_w_u"]
TRI_K56=["parallel_scan_v2","fwd_kernel_o","chunk_o","chunk_fwd_kernel_o"]
def classify(name, which):
    low=name.lower()
    if which=="opus":
        if "gdn_k1" in low: return "k14"
        if "chunk_gated_delta_rule_fwd_h" in low or "gdn_k2" in low or "gdn_wf" in low: return "k56"
    else:
        if any(x in low for x in ["cumsum","prefix_scan","kkt","inverse","recompute_w_u"]): return "k14"
        if "parallel_scan_v2" in low or "fwd_kernel_o" in low or "chunk_o" in low: return "k56"
    return None
def seg(dbdir, which):
    db=glob.glob(dbdir+"/*.db")[0]; con=sqlite3.connect(db)
    tot={"k14":0.0,"k56":0.0}
    for name,dur in con.execute("SELECT name,duration FROM kernels"):
        c=classify(name,which)
        if c: tot[c]+=dur/1000.0
    con.close()
    return tot["k14"]/NITERS, tot["k56"]/NITERS
which=sys.argv[1]; k14,k56=seg(sys.argv[2],which)
print(f"{k14:.1f} {k56:.1f}")
