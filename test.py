import torch
from transformers import RobertaTokenizerFast, MRAForMaskedLM

pytorch_dump_path = "/nobackup/pulijala/model"

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
model = MRAForMaskedLM.from_pretrained(pytorch_dump_path)
model.eval()

text = "Apples grown from seed tend to be very different from those of their parents, and the resultant fruit frequently lacks desired characteristics. Generally, apple cultivars are propagated by clonal grafting onto rootstocks. Apple trees grown without rootstocks tend to be larger and much slower to fruit after planting. Rootstocks are used to control the speed of growth and the size of the resulting tree, allowing for easier <mask>. There are more than 7,500 known cultivars of apples. Different cultivars are bred for various tastes and uses, including cooking, eating raw, and cider production. Trees and fruit are prone to a number of fungal, bacterial, and pest problems, which can be controlled by a number of organic and non-organic means. In 2010, the fruit's genome was sequenced as part of research on disease control and selective breeding in apple production."

inputs = tokenizer(text, return_tensors="pt", padding = 'full_length')

with torch.no_grad(): 
    logits = model(**inputs).logits

mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)

print(predicted_token_id)