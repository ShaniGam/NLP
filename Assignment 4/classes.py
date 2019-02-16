# Represents the candidate we want to classify
class Candidate(object):

    def __init__(self,sent_id, entity1, entity2):
        self._sent_id = sent_id
        self._entity1 = entity1
        self._entity2 = entity2
        self._label = 0
        self._features = []
        self._apposition = False


    @property
    def sent_id(self):
        return self._sent_id

    @property
    def entity1(self):
        return self._entity1

    @property
    def entity2(self):
        return self._entity2

    @property
    def features(self):
        return self._features

    def set_label(self, label):
        self._label = label

    @property
    def apposition(self):
        return self._apposition

    @apposition.setter
    def apposition(self,val):
        self._apposition = val

    def add_feature(self,feature):
        return self._features.append(feature)

    def __str__(self):
        return self._entity1.text + ' , ' + self._entity2.text

    def __repr__(self):
        return 'Candidate(%s,%s)' % (self._entity1.text , self._entity2.text)


# Represents the entity
class Entity(object):

    def __init__(self, entity, text):
        self._entity = entity
        self._text = text

    @property
    def entity(self):
        return self._entity

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self,val):
        self._text = val

