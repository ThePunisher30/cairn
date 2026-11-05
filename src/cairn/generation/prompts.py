import re
from dataclasses import dataclass
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
MARKER = "=== USER ==="
PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")
PROTECTED_TAGS = re.compile(r"<(/?(?:chunk|question|answer|reference))", re.IGNORECASE)


class PromptError(ValueError):
    pass


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: int
    system: str
    user: str

    def placeholders(self):
        return set(PLACEHOLDER.findall(self.system)) | set(PLACEHOLDER.findall(self.user))

    def render(self, **values):
        expected = self.placeholders()
        if set(values) != expected:
            raise PromptError(
                f"prompt {self.name} v{self.version} needs {sorted(expected)}, got {sorted(values)}"
            )

        def fill(text):
            return PLACEHOLDER.sub(lambda match: str(values[match.group(1)]), text)

        return fill(self.system), fill(self.user)


def build_messages(template, question, context):
    system, user = template.render(question=neutralize(question), chunks=format_chunks(context))
    return system, [{"role": "user", "content": user}]


def format_chunks(context):
    parts = []
    for chunk in context.chunks:
        hit = chunk.hit
        source = _attribute(f"{hit.path}:{hit.start_line}-{hit.end_line}")
        parts.append(
            f'<chunk id="{chunk.number}" source="{source}">\n{neutralize(chunk.text)}\n</chunk>'
        )
    return "\n\n".join(parts)


def load_prompt(name, version=None, *, directory=PROMPT_DIR):
    pattern = re.compile(rf"{re.escape(name)}_v(\d+)\.txt")
    found = {}
    for path in Path(directory).glob(f"{name}_v*.txt"):
        match = pattern.fullmatch(path.name)
        if match:
            found[int(match.group(1))] = path
    if not found:
        raise PromptError(f"no prompt named {name!r} in {directory}")
    if version is None:
        version = max(found)
    if version not in found:
        raise PromptError(f"prompt {name!r} has no version {version} (it has {sorted(found)})")
    lines = found[version].read_text(encoding="utf-8").splitlines()
    marks = [i for i, line in enumerate(lines) if line.strip() == MARKER]
    if len(marks) != 1:
        raise PromptError(f"{found[version].name} needs exactly one {MARKER!r} line")
    system = "\n".join(lines[: marks[0]]).strip()
    user = "\n".join(lines[marks[0] + 1 :]).strip()
    return PromptTemplate(name, version, system, user)


def neutralize(text):
    return PROTECTED_TAGS.sub(r"&lt;\1", text)


def _attribute(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace('"', "&quot;")
