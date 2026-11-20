# -*- coding: utf-8 -*-
"""从 Crossref 拉取真实文献元数据并生成 BibTeX（只收录能解析出 DOI 的条目）。"""
import json
import sys
import time
import urllib.parse
import urllib.request

SEL = "DOI,title,author,container-title,issued,volume,issue,page,type,publisher"

# key -> (查询串 或 None, 指定 DOI 或 None)
WANT = [
    ("torrieri1984", None, "10.1109/TAES.1984.310439"),
    ("dijkstra1959", None, "10.1007/BF01386390"),
    ("lozano1979", None, "10.1109/TSE.1979.234169"),
    ("storn1997", None, "10.1023/A:1008202821328"),
    ("nelder1965", None, "10.1093/comjnl/7.4.308"),
    ("gavish1992", "bearing only target location observability Garnier", None),
    ("aoa_localization", "angle of arrival localization accuracy analysis", None),
    ("uav_cooperative", "cooperative localization for multiple UAVs", None),
    ("uav_path_survey", "unmanned aerial vehicle path planning survey", None),
    ("ptz_tracking", "pan tilt zoom camera target tracking", None),
    ("gdop", "geometric dilution of precision positioning geometry analysis", None),
    ("mcmc_sampling", "Monte Carlo statistical methods", None),
    ("evidential", "airborne target geolocation accuracy error propagation", None),
    ("uav_vision", "vision based UAV navigation survey", None),
    ("genetic_scheduling", "multi robot task allocation survey", None),
    ("visibility", "shortest path among polygonal obstacles visibility graph", None),
]


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "paper-builder/1.0 (mailto:none@example.com)"})
    return json.load(urllib.request.urlopen(req, timeout=30))


def fetch(query, doi):
    if doi:
        d = get(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}")
        return d["message"]
    q = urllib.parse.quote(query)
    d = get(f"https://api.crossref.org/works?query.bibliographic={q}&rows=3&select={SEL}")
    items = [i for i in d["message"]["items"]]
    return items[0] if items else None


def fmt_authors(it):
    out = []
    for a in it.get("author", [])[:6]:
        fam = a.get("family", "")
        giv = a.get("given", "")
        if fam:
            out.append(f"{fam}, {giv}" if giv else fam)
        elif a.get("name"):
            out.append(a["name"])
    if len(it.get("author", [])) > 6:
        out.append("others")
    return " and ".join(out) if out else "Unknown"


def clean(t):
    return t.replace("{", "").replace("}", "").replace("&", r"\&").replace("%", r"\%")


def build(key, it):
    title = (it.get("title") or ["?"])[0]
    cont = (it.get("container-title") or [""])[0]
    year = (it.get("issued", {}).get("date-parts") or [[None]])[0][0]
    fields = [
        ("author", fmt_authors(it)),
        ("title", "{" + clean(title) + "}"),
    ]
    if it.get("type") in ("proceedings-article",):
        fields.append(("booktitle", clean(cont)))
    elif cont:
        fields.append(("journal", clean(cont)))
    if year:
        fields.append(("year", str(year)))
    if it.get("volume"):
        fields.append(("volume", str(it["volume"])))
    if it.get("number"):
        fields.append(("number", str(it["number"])))
    if it.get("page"):
        fields.append(("pages", it["page"].replace("-", "--")))
    if it.get("DOI"):
        fields.append(("doi", it["DOI"]))
    kind = "inproceedings" if it.get("type") == "proceedings-article" else "article"
    body = ",\n  ".join(f"{k} = {{{v}}}" for k, v in fields)
    return f"@{kind}{{{key},\n  {body}\n}}\n"


def main():
    out, meta = [], []
    for key, query, doi in WANT:
        try:
            it = fetch(query, doi)
        except Exception as e:  # noqa: BLE001
            print(f"[SKIP] {key}: {type(e).__name__} {e}", file=sys.stderr)
            continue
        if not it:
            print(f"[SKIP] {key}: no item", file=sys.stderr)
            continue
        out.append(build(key, it))
        meta.append({
            "key": key,
            "doi": it.get("DOI"),
            "title": (it.get("title") or [""])[0],
            "venue": (it.get("container-title") or [""])[0],
            "year": (it.get("issued", {}).get("date-parts") or [[None]])[0][0],
            "type": it.get("type"),
        })
        print(f"[OK] {key:18s} {it.get('DOI'):36s} {(it.get('title') or [''])[0][:70]}")
        time.sleep(0.3)

    with open("PAPER_论文/cn/references.bib", "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    with open("PAPER_论文/en/references.bib", "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    with open("PAPER_论文/refs_verified.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\n共 {len(out)} 条，已写入 cn/en references.bib 与 refs_verified.json")


if __name__ == "__main__":
    main()
