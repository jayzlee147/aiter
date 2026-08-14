"""Extract an embedded HSACO from a cached FlyDSL CompiledArtifact."""

import argparse
import glob
import os
import pickle
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    candidates = sorted(
        glob.glob("/root/.flydsl/cache/launch_gdr_decode_kernel_*/*.pkl"),
        key=os.path.getmtime,
        reverse=True,
    )
    for path in candidates:
        with open(path, "rb") as f:
            text = pickle.load(f)._ir_text
        if args.kernel not in text:
            continue
        match = re.search(r'bin = "((?:\\.|[^"\\])*)"', text)
        if match is None:
            continue
        encoded = match.group(1)
        binary = bytearray()
        i = 0
        while i < len(encoded):
            if encoded[i] == "\\":
                pair = encoded[i + 1 : i + 3]
                if len(pair) == 2 and all(c in "0123456789abcdefABCDEF" for c in pair):
                    binary.append(int(pair, 16))
                    i += 3
                else:
                    binary.append(ord(encoded[i + 1]))
                    i += 2
            else:
                binary.append(ord(encoded[i]))
                i += 1
        with open(args.output, "wb") as f:
            f.write(binary)
        print(f"{path}: wrote {len(binary)} bytes to {args.output}")
        return
    raise RuntimeError(f"no cached artifact found for {args.kernel}")


if __name__ == "__main__":
    main()
