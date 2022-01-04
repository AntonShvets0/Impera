def fix_trailing_quotes(text):
    num_quotes = text.count('"')
    if num_quotes % 2 == 0:
        return text
    else:
        return text + '"'

def cut_trailing_sentence(text, allow_action=False):
    text = standardize_punctuation(text)
    last_punc = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
    if last_punc <= 0:
        last_punc = len(text) - 1
    et_token = text.find("<")
    if et_token > 0:
        last_punc = min(last_punc, et_token - 1)
    # elif et_token == 0:
    #     last_punc = min(last_punc, et_token)
    if allow_action:
        act_token = text.find(">")
        if act_token > 0:
            last_punc = min(last_punc, act_token - 1)
        # elif act_token == 0:
        #     last_punc = min(last_punc, act_token)
    text = text[: last_punc + 1]
    text = fix_trailing_quotes(text)
    if allow_action:
        text = cut_trailing_action(text)
    return text

def cut_trailing_action(text):
    lines = text.split("\n")
    last_line = lines[-1]
    if (
        "я сказал" in last_line
        or "Я сказал" in last_line
        or "я сказал" in last_line
        or "Я сказал" in last_line
    ) and len(lines) > 1:
        text = "\n".join(lines[0:-1])
    return text


def standardize_punctuation(text):
    text = text.replace("’", "'")
    text = text.replace("`", "'")
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    return text