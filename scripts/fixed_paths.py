from pathlib import Path
_path = Path(os.path.abspath(__file__))
PUBLIC_REPO_DIR = _path.parent.parent.absolute()
sys.path.append(os.path.join(PUBLIC_REPO_DIR, "scripts"))
print("Using PUBLIC_REPO_DIR = {}".format(PUBLIC_REPO_DIR))