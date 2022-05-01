from transformers import BertJapaneseTokenizer, BertModel
import torch
import json


class SentenceBertJapanese:
    def __init__(self, model_name_or_path, device=None):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path)
        self.model = BertModel.from_pretrained(model_name_or_path)
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def encode(self, sentences, batch_size=8):
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(batch, padding="longest", 
                                           truncation=True, return_tensors="pt").to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

            all_embeddings.extend(sentence_embeddings)

        # return torch.stack(all_embeddings).numpy()
        return torch.stack(all_embeddings)
    

    def sentence_to_vec(self,input_text):
        #sentence_embeddings = model.encode(sentences, batch_size=8)
        sentence_embeddings = self.encode([input_text],batch_size=8)

        #print(sentence_embeddings)
        #return sentence_embeddings.detach().numpy(),type(sentence_embeddings.detach().numpy())

        return sentence_embeddings.detach().numpy().tolist()[0]


"""
def sentence_to_vec(input_text):
    #sentence_embeddings = model.encode(sentences, batch_size=8)
    sentence_embeddings = model.encode([input_text],batch_size=8)

    #print(sentence_embeddings)
    #return sentence_embeddings.detach().numpy(),type(sentence_embeddings.detach().numpy())

    return sentence_embeddings.detach().numpy().tolist()[0]
"""


#MODEL_NAME = "sonoisa/sentence-bert-base-ja-mean-tokens-v2"  # <- v2です。
#model = SentenceBertJapanese(MODEL_NAME)

sentences = [
    "米国防総省のカービー報道官は29日の記者会見で、ウクライナの首都キエフへ進軍していたロシア軍の一部が再配置を始めたと明らかにした。再配置先がウクライナ東部となる可能性に触れて「ロシアはキエフ制圧に失敗した」と断定した。カービー氏は「戦争開始から数日間の迅速な進軍は明らかにキエフが（ロシアにとって）重要な目標だったことを示す」と強調した。侵攻当初からキエフ制圧が目標ではなかったとのロシアの主張に反論した発言だ。", 
    "ロシア国防省は29日にキエフなどで「軍事活動を縮小する」と発表しており、行動が伴うかどうかが焦点になっている。カービー氏は「キエフへの脅威が消えたわけではない」と話し、長距離砲による攻撃が続いていると指摘した。「ロシア軍が（キエフ周辺から）全ての部隊を撤収させるという報道にごまかされてはならない」とも訴え、ロシアの履行状況を見極める方針を示した。",
    "年金生活者らを支援するため検討されている5000円の給付金について、自民党は反対意見が多いなどとして白紙に戻して見直す方針で、政府が来月末までにまとめる緊急対策では、年金生活者らへの支援の在り方も焦点の1つとなる見通しです。",
    "30日午前0時18分ごろ岩手県で震度4の揺れを観測する地震がありました。この地震による津波の心配はありません。震度4の揺れを観測したのは岩手県の宮古市と普代村です。また、震度3の揺れを岩手県の盛岡市、遠野市、釜石市、田野畑村、野田村で観測しました。",
    "新型コロナウイルス禍で３度目の花見シーズンを迎え、静かな花見が定着しつつある。兵庫県内の名所では、サクラにちなんだ催しを３年ぶりに復活させる動きが出てきたが、感染防止のため大人数の宴会などは控えるよう呼び掛けが続く。花見の供は「お酒」より「お茶」という調査結果もあり、世相の変化を裏付ける。",
    "東京ディズニーランド（千葉県浦安市）でショーに出演していた契約社員の女性（４１）が、パワーハラスメントが要因で体調を崩したとして、運営会社のオリエンタルランドを相手取り、慰謝料など計約３３０万円の損害賠償を求めた訴訟の判決が２９日、千葉地裁であった。内野俊夫裁判長は、パワハラを認定しなかったものの、女性が職場で孤立しないよう調整する安全配慮義務に違反したとして、同社に８８万円の支払いを命じた。",
    "30日午前の東京株式市場で日経平均株価は反落し、前日比358円50銭（1.27%）安の2万7893円92銭で終えた。3月期末の配当の権利落ちにより日経平均は240円ほど下押しされた。配当狙いの買い需要がなくなったことで高配当株に売りが出た。ウクライナ情勢の緊張緩和への期待から資源高によるインフレ圧力が低下。米長期金利が低下するなか円高・ドル安が進んだことも重荷となった。"
    ]
#sentence_embeddings = model.encode(sentences, batch_size=8)

"""

from scipy import spatial

def similer_search(sentence_embeddings, query_vec):

    similerity_scores=[]
    for sentence_vec in sentence_embeddings:
        similerity_scores.append(1 - spatial.distance.cosine(sentence_vec, query_vec))
    
    return similerity_scores



#similerity_score.sort(reverse=True)

def dict_sentence_score(sentences,similerity_scores):
    dict_scores_texts = dict(zip(sentences, similerity_scores))
    return dict_scores_texts


#title
st.subheader("News")
st.table(sentences)

text_input = st.text_input("Enter some text", placeholder= "i.e. 10日イオンモールは、名古屋市熱田区の商業施設「イオンモール熱田」で、開業以来初の全面改装を実施すると発表した。今春から秋にかけて、専門店約30店を刷新する。")



if text_input != "":
    st.subheader("input_text📝")
    st.info(text_input)

check = st.button("Text Similer Search🔎")
if check:
    query = [text_input]
    query_vec = model.encode(query, batch_size=8)

    similerity_scores = similer_search(sentence_embeddings, query_vec)
    dict_scores_texts = dict_sentence_score(sentences,similerity_scores)

    st.write("query",query)
    #st.write(dict_scores_texts)

    score_sorted = sorted(dict_scores_texts.items(), key=lambda x:x[1], reverse=True)
    st.write(score_sorted)

"""