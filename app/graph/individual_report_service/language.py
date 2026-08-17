import re


def check_korean_only(report: str) -> list[dict]:
    chinese_text = "".join(sorted(set(re.findall(r"[\u4e00-\u9fff]", report))))
    if chinese_text:
        return [{
            "criterion": 6,
            "description": f"한국어 리포트에 중국어 한자가 포함됨: {chinese_text}",
        }]
    return []
