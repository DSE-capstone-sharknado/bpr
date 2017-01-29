import random
# sampling strategies

class Sampler(object):

    def __init__(self,sample_negative_items_empirically):
        self.sample_negative_items_empirically = sample_negative_items_empirically

    def init(self,data,max_samples=None):
        self.data = data
        self.num_users,self.num_items = data.shape
        self.max_samples = max_samples

    def sample_user(self):
        u = self.uniform_user()
        num_items = self.data[u].getnnz()
        assert(num_items > 0 and num_items != self.num_items)
        return u

    def sample_negative_item(self,user_items):
        j = self.random_item()
        while j in user_items:
            j = self.random_item()
        return j

    def uniform_user(self):
        return random.randint(0,self.num_users-1)

    def random_item(self):
        """sample an item uniformly or from the empirical distribution
           observed in the training data
        """
        if self.sample_negative_items_empirically:
            # just pick something someone rated!
            u = self.uniform_user()
            i = random.choice(self.data[u].indices)
        else:
            i = random.randint(0,self.num_items-1)
        return i

    def num_samples(self,n):
        if self.max_samples is None:
            return n
        return min(n,self.max_samples)

class UniformUserUniformItem(Sampler):

    def generate_samples(self,data,max_samples=None):
        self.init(data,max_samples)
        for _ in xrange(self.num_samples(self.data.nnz)):
            u = self.uniform_user()
            # sample positive item
            i = random.choice(self.data[u].indices)
            j = self.sample_negative_item(self.data[u].indices)
            yield u,i,j

class UniformUserUniformItemWithoutReplacement(Sampler):

    def generate_samples(self,data,max_samples=None):
        self.init(self,data,max_samples)
        # make a local copy of data as we're going to "forget" some entries
        self.local_data = self.data.copy()
        for _ in xrange(self.num_samples(self.data.nnz)):
            u = self.uniform_user()
            # sample positive item without replacement if we can
            user_items = self.local_data[u].nonzero()[1]
            if len(user_items) == 0:
                # reset user data if it's all been sampled
                for ix in self.local_data[u].indices:
                    self.local_data[u,ix] = self.data[u,ix]
                user_items = self.local_data[u].nonzero()[1]
            i = random.choice(user_items)
            # forget this item so we don't sample it again for the same user
            self.local_data[u,i] = 0
            j = self.sample_negative_item(user_items)
            yield u,i,j

class UniformPair(Sampler):

    def generate_samples(self,data,max_samples=None):
        self.init(data,max_samples)
        for _ in xrange(self.num_samples(self.data.nnz)):
            idx = random.randint(0,self.data.nnz-1)
            u = self.users[self.idx]
            i = self.items[self.idx]
            j = self.sample_negative_item(self.data[u])
            yield u,i,j

class UniformPairWithoutReplacement(Sampler):

    def generate_samples(self,data,max_samples=None):
        self.init(data,max_samples)
        idxs = range(self.data.nnz)
        random.shuffle(idxs)
        self.users,self.items = self.data.nonzero()
        self.users = self.users[idxs]
        self.items = self.items[idxs]
        self.idx = 0
        for _ in xrange(self.num_samples(self.data.nnz)):
            u = self.users[self.idx]
            i = self.items[self.idx]
            j = self.sample_negative_item(self.data[u])
            self.idx += 1
            yield u,i,j

class ExternalSchedule(Sampler):

    def __init__(self,filepath,index_offset=0):
        self.filepath = filepath
        self.index_offset = index_offset

    def generate_samples(self,data,max_samples=None):
        self.init(data,max_samples)
        f = open(self.filepath)
        samples = [map(int,line.strip().split()) for line in f]
        random.shuffle(samples)  # important!
        num_samples = self.num_samples(len(samples))
        for u,i,j in samples[:num_samples]:
            yield u-self.index_offset,i-self.index_offset,j-self.index_offset
