import os

import torch


def save_checkpoint(net, clf, critic, epoch, args, file_name):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'clf': clf.state_dict(),
        'critic': critic.state_dict(),
        'epoch': epoch,
        'args': vars(args)
    }
    if not os.path.isdir('~/efficient-contrastive-learning/checkpoint'):
        os.mkdir('~/efficient-contrastive-learning/checkpoint')
    destination = os.path.join('~/efficient-contrastive-learning/checkpoint', f"{file_name}.pth")
    torch.save(state, destination)