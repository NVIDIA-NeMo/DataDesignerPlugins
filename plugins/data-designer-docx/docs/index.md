# data-designer-docx

Renders Data Designer rows as Microsoft Word documents. Registers a `docx`
processor that writes one `.docx` per row and records each file's path back into
the dataset, so rows and documents stay joined.

## Installation

```bash
uv add data-designer data-designer-docx
```

## How it works

The plugin ships a Pydantic model, `WordDocument`, that is used twice: as the
`output_format` of an LLM structured column, and as the input contract of the
renderer. Because both ends share one definition, a model that emits something
unrenderable produces a validation error on the column — which Data Designer
already knows how to retry — instead of a parsing failure downstream.

That is also why the package contains no markdown parsing. The LLM generates the
document's *structure*, and the renderer walks it.

```text
llm-structured(document: WordDocument)  ->  processor(docx)  ->  documents/<name>/*.docx
```

## Configuration

| Field | Required | Description |
| --- | --- | --- |
| `name` | Yes | Processor name; also the output subfolder. |
| `document_column` | Yes | Column holding a `WordDocument`-shaped value. |
| `output_subdir` | No | Folder under the dataset directory. Defaults to `documents`. Must be relative and must not name a Data Designer-managed directory. |
| `filename_template` | No | Jinja2 template for the file name. Defaults to `document.docx`, which references no dataset columns; duplicates get a numeric suffix. |
| `output_path_column` | No | Column receiving the written path. Defaults to `docx_path`. |
| `metadata_columns` | No | Label to Jinja2 template pairs, rendered as a front-matter table. |
| `core_property_columns` | No | Word core property (`author`, `category`, `subject`, `keywords`) to Jinja2 template pairs. |
| `template_path` | No | A `.docx` supplying corporate styles, header, and footer. |
| `footer_template` | No | Jinja2 template for the page footer, applied to every section. |
| `table_style` | No | Table style name; must exist in the template. Defaults to `Table Grid`. |
| `number_sections` | No | Prefix section headings with `1.`, `2.`, and so on. Defaults to `True`. |

## Implementation notes

**Stage choice.** The processor implements `process_after_batch` rather than
`process_after_generation`. Documents then stream out while the run is still in
progress, the row count stays fixed as the async engine requires at that stage,
and the dataset stays resumable — `process_after_generation` rewrites the final
parquet and marks the dataset terminal for resume.

**Output location.** Documents are written to `<output_subdir>/<name>/`, never
under `processors-files/`. Both components are validated as contained, relative,
non-reserved path segments, and the resolved directory is asserted to sit beneath
the dataset directory before anything is written. Data Designer reads every directory there back as a
parquet dataset, so `.docx` files placed there make `preview()` fail with
*"Parquet magic bytes not found in footer"*. Binary artifacts get their own
folder, the same way generated images live under `images/`. The config validator
rejects the reserved names.

**Ragged tables.** Structured outputs constrain the shape of the JSON, not the
arithmetic inside it — a model asked for a three-column table will occasionally
return a row with two cells. Rows are padded or truncated to the header width
rather than triggering a retry, which would cost a whole document generation.

**Package layout.** `schema.py` holds the contract, `render.py` is pure
python-docx with no Data Designer imports (so layout can be iterated without
spending tokens), `config.py` is user-facing, and `impl.py` is engine-side.

## Templates

`template_path` should point at a `.docx` containing styles, header, and footer
but **no body content** — python-docx appends generated content after anything
already in the file, so a template with a cover page yields a cover page on every
document.

**Structured values are not re-decoded.** Data Designer's recursive JSON decoding
rewrites string leaves that look like scalars, turning `"30"` into `30` and
`"true"` into `True` — precisely the values a key-data table carries, and values
the schema then rejects. The processor therefore validates the document column
from the raw record and only JSON-decodes it when the top-level value is a
string. The recursively decoded copy is used for Jinja templates only.

**Resume safety.** File name collisions are tracked with case-folded keys, since
macOS and Windows treat `A.docx` and `a.docx` as the same file, and the set is
seeded from documents already on disk. A resumed run therefore cannot overwrite a
document written by a batch that completed before the resume.

**Footers.** Generated content is appended after any body content the template
already has, so it lands in the template's final section. `footer_template` is
applied to every section rather than just the first, which would otherwise leave
the generated pages showing the template's own footer.
