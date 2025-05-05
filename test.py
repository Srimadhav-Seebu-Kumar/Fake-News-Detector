import torch
import torch.nn as nn
import pickle

with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, hidden_dim=256, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)
        x = self.fc(x)
        return self.sigmoid(x).squeeze(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TransformerClassifier(vocab_size=len(vocab))
model.load_state_dict(torch.load("transformer_fake_news_classifier.pth", map_location=device))
model.to(device)
model.eval()

def tokenize(text):
    return text.lower().split()

def encode(text, max_len=512):
    tokens = tokenize(text)
    enc = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    if len(enc) > max_len:
        enc = enc[:max_len]
    else:
        enc += [vocab["<PAD>"]] * (max_len - len(enc))
    return torch.tensor(enc, dtype=torch.long).unsqueeze(0).to(device)  

text = "New York governor questions the constitutionality of federal tax overhaul, NEW YORK/WASHINGTON (Reuters) - The new U.S. tax code targets high-tax states and may be unconstitutional, New York Governor Andrew Cuomo said on Thursday, saying that the bill may violate New York residents’ rights to due process and equal protection.  The sweeping Republican tax bill signed into law by U.S. President Donald Trump on Friday introduces a cap, of $10,000,  on deductions of state and local income and property taxes, known as SALT. The tax overhaul was the party’s first major legislative victory since Trump took office in January.  The SALT provision will hit many taxpayers in states with high incomes, high property values and high taxes, like New York, New Jersey and California. Those states are generally Democratic leaning.  “I’m not even sure what they did is legally constitutional and that’s something we’re looking at now,” Cuomo said in an interview with CNN. In an interview with CNBC, Cuomo suggested why the bill may be unconstitutional.  “Politics does not trump the law,” Cuomo said on CNBC. “You have the constitution, you have the law, you have due process, you have equal protection. You can’t use politics just because the majority controls to override the law.” The Fifth Amendment of the Constitution, better known for its protection against self-incrimination, also protects individuals from seizure of life, liberty or property without due process and has been interpreted by the Supreme Court as guaranteeing equal protection by the law.  Cuomo and California Governor Jerry Brown, both Democrats, have previously said they were exploring legal challenges to SALT deduction limits.  Law professors have said legal challenges would likely rest on arguing that the provision interferes with the protection of states’ rights under the U.S. Constitution’s 10th Amendment. Tax attorneys said Cuomo’s legal argument against the tax bill could be that it discriminates and places an unjust tax burden on states that heavily voted for Democrats in the past - known as “blue states.” “The de facto effect of this legislation is to discriminate against blue states and particularly from (Cuomo’s) perspective the state of New York,” said Joseph Callahan, an attorney with the law firm Mackay, Caswell & Callahan in New York.  But some tax experts noted the U.S. Supreme Court has interpreted the 16th Amendment to give Congress broad latitude to tax as it sees fit. In a frequently cited 1934 decision, the Supreme Court called tax deductions a “legislative grace” rather than a vested right. “I don’t understand how they think they have a valid lawsuit here,” David Gamage, a professor of tax law at Indiana University’s Maurer School of Law, told Reuters last week, speaking generally about governors in blue states that could challenge the tax bill.  Cuomo also said on Thursday that New York is proposing a restructuring of its tax code. He provided no details.  A group of 13 law professors on Dec. 18 published a paper suggesting ways that high-tax states could minimize the effects of the SALT deduction cap.  Their suggestions included shifting more of the tax burden onto businesses in the form of higher employer-side payroll taxes, since the federal tax bill’s cap on SALT deductions only applies to individuals and not businesses. States also could raise taxes on pass-through entities, which the federal tax bill specifically benefits with a lower rate on a portion of their income. On Friday, Cuomo said he would allow state residents to make a partial or full pre-payment on their property tax bill before Jan. 1, allowing taxpapyers to deduct such payments for 2017 before the cap kicks in, prompting a wave of residents to pay early.  However, the U.S. Internal Revenue Service on Wednesday advised homeowners that the pre-payment of 2018 property taxes may not be deductible. "
encoded = encode(text)
with torch.no_grad():
    pred = model(encoded).item()
    print("Prediction (probability of being TRUE):", pred)

