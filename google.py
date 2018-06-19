from googletrans import Translator
translator = Translator(service_urls=
                        ['translate.google.cn'])
# translations = \
#     translator.\
#         translate(['为何本月花呗不能分期',
#                    '分期付款花呗是不是给钱商家'],
#                   src='zh-CN', dest='en')
#
# for translation in translations:
#     print(translation.origin,'->',translation.text)

def trans_en(txt):
    return translator.translate(txt,src='en',dest='zh-CN').text

def trans_zh(txt):
    return translator.translate(txt,src='zh-CN',dest='en').text

def trans_back(txt):
    return trans_en(trans_zh(txt))

# print(trans_back('分期付款花呗是不是给钱商家'))