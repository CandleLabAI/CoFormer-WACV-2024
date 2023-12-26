from ai4bharat.transliteration import XlitEngine

def translitrate(text, tgt_languages=["hi"]):
  out = e.translit_word(text, topk=5) 
  return [out[lang][0].rstrip('\u200c') for lang in tgt_languages]

if __name__ == "__main__":
    
  tgt_languages = ["hi", "bn", "ta", "kn", "te"]
    
  e = XlitEngine(tgt_languages, beam_width=10, rescore=True)
  
  example_inputs = [
    'School',
    'Photosynthesis',
    'Chlorophyll',
    'Google',
    'Transformer',
    'Neural',
    'Samsung',
    'Pizza',
    'Burger',
    'Internet',
    'Macbook',
    'Algorithms',
    'ChatGPT',
    'Integrals',
    'neural networks',
    'diffuson',
    'Nightline',
    'Ethanol',
    'Ethene',
    'Ethyne',
    'Fossil',
    'Jubilee',
    'Onkar',
    'Prajwal',
    'C.B.S.E',
    'CBSE',
  ]
  
  out = {text: translitrate(text, tgt_languages) for text in example_inputs}

  with open('./transliterate_out.txt', 'w', encoding='utf-8') as fp:
    fp.writelines(f'{k} : {v}'+'\n' for k,v in out.items())