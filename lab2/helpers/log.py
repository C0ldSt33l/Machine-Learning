from datetime import datetime

LOG_DIR = r"log/"


def write_log(log: str, filename: str | None = None, extention: str = "log"):
    if filename is None:
        now = datetime.now()
        filename = now.strftime("%Y-%m-%d %H_%M_%S")
    filename += "." + extention

    with open(LOG_DIR + filename, "a") as f:
        f.write(log)


def append_tab_to_multilines(lines: str, count: int = 1) -> str:
    return "\n".join("\t" * count + l for l in lines.splitlines())
