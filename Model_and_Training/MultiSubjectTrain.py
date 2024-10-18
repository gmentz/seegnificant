import os
import torch.utils.data as data
import time
import wandb
import torch
from .model import seegnificant
from .sEEGDataset import ParticipantDataset
from .utils import train, evaluate, reset_weights
os.environ["WANDB_SILENT"] = "true"
import itertools


training_hparams = dict(
    seed=0,
    epochs=1000,
    batch_size=64,
    train_frac=0.7,
    val_frac=0.15,
    test_frac=0.15,
    num_participants=3,
    lr=0.001,
    b1=0.5,
    b2=0.999,
    gamma=0.9,
    step_size=100,
)
model_hparams = dict(
    latent_dim=2,
    num_blocks=1,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)

wandb.login()

for name in ['Synth_' + x for x in ['1', '2', '3']]:  # This specified the name of the left-out participant

    config = {**training_hparams, **model_hparams}

    print('=' * 97)
    print('Training on all participant except: ', name)

    with wandb.init(group=name, project="Seegnificant", config=config, name='Train_Backbone_' + name):

        config = wandb.config

        g = torch.Generator().manual_seed(config.seed)

        # Load the dataset and split into training and validation
        dataset = torch.load(os.getcwd() + '/data/dataset.pt')
        training_data, validation_data, testing_data = torch.utils.data.random_split(dataset, [config.train_frac,  config.val_frac, config.test_frac], generator=g)

        # Keep data from all participants but the Left out Participant
        training_data.indices = list(itertools.compress(training_data.indices, training_data.dataset.id[training_data.indices] != training_data.dataset.participants.index(name)))
        validation_data.indices = list(itertools.compress(validation_data.indices, validation_data.dataset.id[validation_data.indices] != validation_data.dataset.participants.index(name)))
        testing_data.indices = list(itertools.compress(testing_data.indices, testing_data.dataset.id[testing_data.indices] != testing_data.dataset.participants.index(name)))

        # Check that there is no overlap between training/validation/testing
        assert set(training_data.indices).intersection(validation_data.indices) == set()
        assert set(training_data.indices).intersection(testing_data.indices) == set()
        assert set(validation_data.indices).intersection(testing_data.indices) == set()

        trainloader = torch.utils.data.DataLoader(training_data, batch_size=config.batch_size, shuffle=True, generator=g)
        validloader = torch.utils.data.DataLoader(validation_data, batch_size=config.batch_size, shuffle=True, generator=g)
        testloader = torch.utils.data.DataLoader(testing_data, batch_size=config.batch_size, shuffle=True, generator=g)

        # Initialize the models
        model = seegnificant(config.latent_dim, config.num_blocks, config.num_participants).to(device)
        reset_weights(model)

        # Initialize the optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], betas=(config['b1'], config['b2']))
        criterion_reg = torch.nn.HuberLoss()

        # Make the learning rate decay
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=config.step_size, gamma=config.gamma)

        # Compute training/validation metrics
        best_val_loss_reg = None
        val_loss_reg = None

        # Train the network on all subjects but one
        for epoch in range(1, config.epochs + 1):
            epoch_start_time = time.time()
            model, train_loss_reg, train_r2 = train(model, trainloader, optimizer, criterion_reg, epoch, device)
            lr_scheduler.step()
            print('| Epoch {:3d} | time: {:5.2f}s | Train | Loss {:5.5f} | R2 {:1.2f} '.format(
                epoch, (time.time() - epoch_start_time), train_loss_reg, train_r2))

            if epoch % 10 == 0:
                epoch_start_time = time.time()
                val_loss_reg, valid_r2 = evaluate(model, validloader, criterion_reg, epoch, device)

                if not best_val_loss_reg or val_loss_reg < best_val_loss_reg:
                    torch.save(model.state_dict(), os.getcwd() + '/models/MultiSubjectModelWithout' + name)
                    best_val_loss_reg = val_loss_reg
                print('-' * 97)
                print('| Epoch {:3d} | time: {:5.2f}s | Valid | Loss {:5.5f} | R2 {:1.2f}'.format(
                    epoch, (time.time() - epoch_start_time), val_loss_reg, valid_r2))
                print('-' * 97)

        # Evaluate the "best" regression model
        checkpoint = torch.load(os.getcwd() + '/models/MultiSubjectModelWithout' + name)
        model.load_state_dict(checkpoint)

        val_loss_reg, valid_r2 = evaluate(model, validloader, criterion_reg, epoch, device)
        test_loss_reg, test_r2 = evaluate(model, testloader, criterion_reg, epoch, device)
        print('| End of training | ValLoss {:5.5f} | ValR2{:1.2f} | TestLoss {:5.5f} | TestR2 {:1.2f}'.format(
            val_loss_reg, valid_r2, test_loss_reg, test_r2))
        print('=' * 97)



