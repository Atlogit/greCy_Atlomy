from cassis import *
import zipfile
import os
import unicodedata
from datetime import date
import spacy
from spacy.tokens import DocBin

import unicodedata
def clean_text(text: str) -> str:
    #Cleans the given text by stripping accents and lowercasing.
    try:
        non_accent_characters = [
        char for char in unicodedata.normalize('NFKD', text)
        if unicodedata.category(char) != 'Mn'
        #or char == '̓'  # Greek coronis
        ]
    # str.lower() works for unicode characters
        return ''.join(non_accent_characters).lower()
    except TypeError:
        return text

# create blank nlp object
# "grc_ud_proiel_trf" is the name of the spaCy models pipeline for ancient Greek NLP. Created by Jacobo Myerston, with the aristoBERTo tranformer model.
# "grc" is the spaCy default Greek language model
nlp = spacy.load("/root/Projects/Atlomy/git/greCy_ATLOMY/training/Lemmatize_transformer_morefiles/lemmatizer/model-best")

# iterate through zipped files in path
absolute_path = os.path.dirname(__file__)
OriginPATH = os.path.join(absolute_path, 'assets/NER_assets/INCEpTION_files')
TargetPath = os.path.join(absolute_path, 'corpus')

for filename in os.listdir(OriginPATH):
    if filename.endswith(".zip"):
        print(filename)
        with zipfile.ZipFile(OriginPATH + "//" + filename, 'r') as zip_ref:
            zip_ref.extractall(TargetPath)

path_to_cas = TargetPath
with open("{0}/TypeSystem.xml".format(path_to_cas), 'rb') as f:
    typesystem = load_typesystem(f)

# find file name with xmi extension and load it
for file in os.listdir(path_to_cas):
    if file.endswith(".xmi"):
        print("FILENAME:"+file)
        with open("{0}/{1}".format(path_to_cas, file), 'rb') as f:
            cas = load_cas_from_xmi(f, typesystem=typesystem)

# Load all sentences and tokens into spacy doc object
# create DocBin object
        db = DocBin()
        lemma_dict = {} #type: Dict[str, Set[str]]
        for sentence in cas.select("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"):
            #remove accents
            #sentence = clean_text(sentence.get_covered_text())
            doc = nlp.make_doc(clean_text(sentence.get_covered_text()))
            #doc = nlp.make_doc(sentence)
            ents = []
            for token in cas.select_covered('webanno.custom.CategoryType', sentence):
                # if token Value is not empty and token is not a space
                if token.value("Value") is not None and token is not None:
                    # print covered text and value
                    print(token.get_covered_text(), token.value("Value"))
                    # create a span with the token start and end and the label
                    # find begin and end position of the token relative to the sentence
                    start = token.begin - sentence.begin
                    end = token.end - sentence.begin
                    span = doc.char_span(start, end, label=clean_text(token.value("Value")), alignment_mode='expand')
                    print(span)
                    # add the span to the entities
                    ents.append(span)
            # add the token and the lemma annotation to the lemmata dictionary
            for token in cas.select_covered('de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Lemma', sentence):
                lemma_dict[token.get_covered_text()] = token.value

            # create a doc object from the doc and add the entities
            doc.set_ents(ents)
            print (doc.ents)
            print("DONEEEEEE")
            # add the doc to the docbin
            db.add(doc)
            print("TOTAL:",len(db))
        # save session data to disk
        db.to_disk(path_to_cas + "/INCEpTION_Data/" + "{0}_data.spacy".format(file[:-4]))
        f.close()
        # save lemma dictionary to disk as txt file
        #lemma_dict = {unicodedata.normalize('NFKD', key): unicodedata.normalize('NFKD', value) for key, value in lemma_dict.items()}
        lemma_dict = {clean_text(key): clean_text(value) for key, value in lemma_dict.items()}

        with open(path_to_cas + "/INCEpTION_Data/" + "{0}_lemma_dict.txt".format(file[:-4]), "w", encoding="utf-8") as f:
            for key, value in lemma_dict.items():
                f.write(key + "\t" + value + "\n")
        f.close()
       

        # split the data into train and test without split function
        # get_docs returns a generator
        #docs = list(db.get_docs(nlp.vocab))
        #print(docs)
        #train_data = docs[:int(len(docs)*0.8)]
        #test_data = docs[int(len(docs)*0.8):]

        # Split the data in 60:20:20 for train:dev:test dataset
        # save the docbin objects

        from sklearn.model_selection import train_test_split
        TRAIN_DATA = list(db.get_docs(nlp.vocab))
        # TRAIN_DATA shape
        #print TRAIN_DATA shape
        #print (len(TRAIN_DATA))
        #print(TRAIN_DATA)
        print(len(TRAIN_DATA))
        
        if len(TRAIN_DATA) == 1:
            db_train = DocBin(docs=TRAIN_DATA)
            db_train.to_disk(path_to_cas + "/train/ner_train/" + "{0}_train.spacy".format(file[:-4]))
        else:
            train_data, test_data = train_test_split(TRAIN_DATA, test_size=0.2, random_state=42)
            db_train = DocBin(docs=train_data)
            db_train.to_disk(path_to_cas + "/train/ner_train/" + "{0}_train.spacy".format(file[:-4]))   
        
        try:
            test_data
            if len(test_data) <= 2:
                print("not enough for evaluation test")
                db_dev = DocBin(docs=test_data)
                db_dev.to_disk(path_to_cas + "/dev/ner_dev/" + "{0}_dev.spacy".format(file[:-4]))
                del test_data
            else:
                dev_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)
                db_dev = DocBin(docs=dev_data)
                db_dev.to_disk(path_to_cas + "/dev/ner_dev/" + "{0}_dev.spacy".format(file[:-4]))
                db_test = DocBin(docs=test_data)
                db_test.to_disk(path_to_cas + "/test/ner_test/" + "{0}_test.spacy".format(file[:-4]))
                del dev_data, test_data
        except NameError:
            print("Only train data was created")
            del train_data
            
