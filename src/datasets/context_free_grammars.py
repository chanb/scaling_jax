import numpy as np


class CFGSampler:
    def __init__(
        self,
        production_rules,
        entry_key,
        seed,
        production_probs=None,
    ):
        if production_probs is None:
            production_probs = dict()
            for k, v in production_rules.items():
                production_probs[k] = [1 / len(v)] * len(v)
        else:
            for _, probs in production_probs.items():
                probs = np.array(probs)
                assert np.all(probs > 0.0)
                assert np.all(probs < 1.0) or len(probs) == 1

        self.production_rules = production_rules
        self.entry_key = entry_key
        self.production_probs = production_probs
        self.seed = seed
        self._rng = np.random.RandomState(seed)

    def sample(self):
        def _sample(curr_rule_key):
            curr_sentence = ""

            if curr_rule_key not in self.production_rules:
                return curr_rule_key

            sampled_entry = self._rng.choice(
                self.production_rules[curr_rule_key],
                p=self.production_probs[curr_rule_key],
            )

            tokens = sampled_entry.split(" ")
            if len(tokens) > 1 or tokens[0] in self.production_rules:
                # recursive call
                for token in tokens:
                    curr_sentence += " {}".format(_sample(token))
                return curr_sentence[1:]
            else:
                # base case
                return sampled_entry
            
        sampled_seq = _sample(self.entry_key).split(" ")
        return [token for token in sampled_seq if token != ""]
    
    # TODO: Check if an input is in the language


if __name__ == "__main__":
    production_rules = {
        "seq": ["<BOS> expr <EOS>"],
        "atom": [str(el) for el in range(10)],
        # "comp": ["atom < atom", "atom > atom", "atom = atom"],
        "comp": ["atom SWAP atom"],
        "noop": ["noop NOOP", ""],
        "eval": ["EVAL PLACEHOLDER"],
        "inc": ["inc INC", ""],
        "op": ["op noop inc noop + noop atom noop inc noop eval noop", ""],
        "stmt": ["noop atom noop op", "noop atom noop op <SEP> stmt"],
        "expr": ["noop comp noop <SEP> expr", "stmt"],
    }
    production_probs = {
        "seq": [1.0],
        "atom": [0.1] * 10,
        # "comp": [1/3, 1/3, 1/3],
        "comp": [1.0],
        "noop": [0.1, 0.9],
        "eval": [1.0],
        "inc": [0.1, 0.9],
        "op": [0.5, 0.5],
        "stmt": [0.8, 0.2],
        "expr": [0.6, 0.4],
    }
    entry_key = "seq"
    seed = 42

    sampler = CFGSampler(production_rules, entry_key, seed, production_probs)

    for _ in range(20):
        print(" ".join(sampler.sample()))
