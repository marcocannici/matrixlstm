import re
import torch


class Scheduler:
    def __init__(self, network, param_names):

        self.network = network
        self.param_names = param_names
        self.check_param_names()
        self.last_epoch = -1

    def get_leaves(self, attr_hierarchy):
        # Compile a regex to match lists
        list_attr = re.compile(r"^(\w+)\[([\*|\d+])\]$")
        # 'attr' may be something like "conv1.weight"
        # or even layer[0].weight and layer[*].weight
        # We split the tree structure first
        attrs = attr_hierarchy.split('.')
        leaves = [self.network]
        # and make sure that the last level is not a list
        if list_attr.findall(attrs[-1]):
            raise ValueError('Invalid name, the last level cannot be a list')
        # We now apply getattr till the last level
        for attr in attrs[:-1]:
            new_leaves = []
            for leaf in leaves:
                # If it is a list
                list_match = list_attr.findall(attr)
                if list_match:
                    list_name = list_match[0][0]
                    list_index = list_match[0][1]
                    child_list = getattr(leaf, list_name)
                    if list_index != '*':
                        new_leaves += [child_list[int(list_index)]]
                    else:
                        new_leaves += child_list
                else:
                    new_leaves += [getattr(leaf, attr)]
            leaves = new_leaves

        return leaves, attrs[-1]

    def state_dict(self):
        return {'last_epoch': self.last_epoch,
                'param_names': self.param_names,
                'param_states': self.get_params_state()}

    def load_state_dict(self, state_dict):
        self.last_epoch = state_dict['last_epoch']
        self.param_names = state_dict['param_names']
        self.set_params_state(state_dict['param_states'])

    def check_param_names(self):
        # Call get_leaves on all names, if it fails let the exception raise
        for name in self.param_names:
            self.get_leaves(name)

    def get_params_state(self):
        state = []
        for name in self.param_names:
            param_state = []
            leaves, attr_name = self.get_leaves(name)
            for leaf in leaves:
                param_state.append(getattr(leaf, attr_name))
            state.append(param_state)
        return state

    def set_params_state(self, param_states):
        for name, states in zip(self.param_names, param_states):
            leaves, attr_name = self.get_leaves(name)
            for leaf, state in zip(leaves, states):
                attr = getattr(leaf, attr_name)
                if isinstance(attr, torch.Tensor):
                    attr.fill_(state)
                else:
                    setattr(leaf, attr_name, state)

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        new_state = self.next_state()
        if new_state:
            self.set_params_state(new_state)

    def next_state(self):
        raise NotImplementedError


class LambdaScheduler(Scheduler):

    def __init__(self, network, param_names, lambdas):
        super().__init__(network, param_names)
        self.lambdas = lambdas

    def next_state(self):
        state = self.get_params_state()
        for attr_state, attr_lmbd in zip(state, self.lambdas):
            for i, attr in enumerate(attr_state):
                attr_state[i] = attr_lmbd(attr, self.last_epoch)
        return state


class StepScheduler(Scheduler):

    def __init__(self, network, param_names, step_size, gammas=0.1):
        super().__init__(network, param_names)
        self.step_size = step_size
        self.gammas = self._extend(gammas, len(param_names))

    @staticmethod
    def _extend(val, length):
        if isinstance(val, list):
            assert len(val) == length
        else:
            val = [val] * length
        return val

    def state_dict(self):
        state = super().state_dict()
        state.update({'step_size': self.step_size})
        state.update({'gammas': self.gammas})
        return state

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.step_size = state_dict['step_size']
        self.gammas = state_dict['gammas']

    def next_state(self):
        if self.last_epoch != 0 and self.last_epoch % self.step_size == 0:
            state = self.get_params_state()
            for attr_state, attr_gamma in zip(state, self.gammas):
                for i, attr in enumerate(attr_state):
                    attr_state[i] = attr * attr_gamma
            return state


if __name__ == "__main__":

    class Inner:
        def __init__(self, b):
            self.b = b

    class Test:
        def __init__(self, a1, a2, b):
            self.a1 = a1
            self.a2 = a2
            self.l = [Inner(b) for _ in range(2)]

    t = Test(1, 2, 3)
    s = StepScheduler(t, ["a1", "a2", "l[*].b"], 1)
    print(f"{t.a1} {t.a2} [{t.l[0].b}, {t.l[1].b}]")
    s.step()
    s.step()
    print(f"{t.a1} {t.a2} [{t.l[0].b}, {t.l[1].b}]")
