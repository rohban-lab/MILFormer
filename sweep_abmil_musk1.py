import wandb


def get_sweep_conf():
    sweep_configuration = {
        'method': 'grid',
        'name': 'ABMIL on Musk1 & Musk2',
        'program': 'train.py',
        'description': '',
        'parameters': {
            'dataset': {
                'values': ['musk1', 'musk2']
            },
            'arch': {
                'values': ['abmil-paper', 'abmil']
            },
            'cv_current_fold': {
                'values': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            },
            'seed': {  # Each combination should be run {seed} times
                'values': [1]
            },
        }
    }
    return sweep_configuration


def main():
    config = get_sweep_conf()
    wandb.sweep(
        config,
        project=f"MIL_camelyon16",
    )


if __name__ == '__main__':
    main()
