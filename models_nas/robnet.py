from .basic_model import Network


def robnet(genotype_list, share=False, **kwargs):
    return Network(genotype_list=genotype_list, share=share, **kwargs)
