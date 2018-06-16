from sogou_translate import SogouTranslate, SogouLanguages

translator = SogouTranslate('5328612ac485498d8c6b0a5e666126ef',
                       '10daaa0d8ba608ef942a7e64be5676d5')

def trans_en(en_text):
    zh_text = translator.translate(en_text,
                              from_language=SogouLanguages.EN,
                              to_language=SogouLanguages.ZH_CHS)
    return zh_text

def trans_zh(zh_text):
    en_text = translator.translate(zh_text,
                              from_language=SogouLanguages.ZH_CHS,
                              to_language=SogouLanguages.EN)
    return en_text

def trans_back(zh_text):
    return trans_en(trans_zh(zh_text))

# print(trans_zh('就是我花呗忘记还款了。逾期一天。有事吗'))
# print(trans_back('借呗开通需要什么资料'))
# print(trans_zh('借呗开通需要什么资料'))
# print(trans_en('What information do you need for opening'))

# print(trans_back('花呗系统繁忙'))

