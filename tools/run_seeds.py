import argparse
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", default="config_dev.yaml")
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    args = p.parse_args()

    for i, s in enumerate(args.seeds, start=1):
        print(f"=== run {i}/{len(args.seeds)} seed={s} ===", flush=True)
        subprocess.check_call([
            "python",
            "src/train.py",
            "--config",
            args.config,
            "--seed",
            str(s),
        ])


if __name__ == "__main__":
    main()
