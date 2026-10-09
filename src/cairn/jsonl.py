import json
from collections.abc import Iterable, Iterator
from pathlib import Path


class JsonlError(Exception):
    pass


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            json_line = json.dumps(row, ensure_ascii=False)
            f.write(json_line + '\n')
            count += 1
    return count


def read_jsonl(path: Path) -> Iterator[dict]:
    path = Path(path)
    with open(path, 'r',encoding='utf-8') as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise JsonlError(f"{path.name}: line {line_number}: {e}") from e
            yield row