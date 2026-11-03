import argparse, pathlib, shutil, importlib.metadata as md

FILES = ["app.py", "src/__init__.py", "src/contract_transform.py", "src/treat.py", "src/features.py", "src/predict.py", "src/demo.py",
         "models/pd_model.joblib", "contract/transform.joblib", "contract/data_spec.json"]
PKGS = ["pandas", "numpy", "scipy", "scikit-learn", "lightgbm", "joblib", "pyarrow"]
FORBIDDEN = ("data/", "reference_transforms", "analytical_dataset", "holdout", ".csv", ".parquet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gradio", required=True)
    ap.add_argument("--python", default="3.12")
    a = ap.parse_args()
    out = pathlib.Path("space_build")
    if out.exists():
        shutil.rmtree(out)
    for f in FILES:
        dst = out / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, dst)
    (out / "requirements.txt").write_text("\n".join(f"{p}=={md.version(p)}" for p in PKGS) + "\n")
    front = (f"---\ntitle: Credit risk PD demo\nsdk: gradio\nsdk_version: {a.gradio}\npython_version: \"{a.python}\"\n"
             "app_file: app.py\npinned: false\n---\n\n")
    note = ("\n\n## About this demo\nThe Space scores one invented or typed-in applicant through the same `predict.py` used by the tests. "
            "It contains the trained model, the fitted transform and the field specification, and no loan data of any kind.\n")
    (out / "README.md").write_text(front + open("model_card.md", encoding="utf-8").read() + note, encoding="utf-8")
    bad = [p.as_posix() for p in out.rglob("*") if p.is_file() and any(s in p.as_posix().lower() for s in FORBIDDEN)]
    assert not bad, f"data files in the Space folder: {bad}"
    print("built space_build/:", sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()))


if __name__ == "__main__":
    main()