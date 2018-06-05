import numpy as np

def combine_split_result(s):
    """Combines the result of a re.split splitting
    with the splitters retained. E.g. ['I', '/', 'am']
    -> ['I', '/am'].
    """
    if len(s) > 1:
        ss = [s[0]]
        for i in range(1, len(s)//2+1):
            ii = i-1
            ss.append(s[i] + s[i+1])
        return ss
    else:
        return s

def cosine_similarity(a, b):
        """Takes 2 vectors a, b and returns the cosine similarity according 
        to the definition of the dot product
        """
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if np.isclose(norm_a, 0) or np.isclose(norm_b, 0):
            return 0
        else:
            return dot_product / (norm_a * norm_b)
