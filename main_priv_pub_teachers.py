import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import ToTensor
import os
from models.net_mnist import CNN
import torch
from aggregate_teachers import prepare_dataloader_to_label, ask_teachers_to_label_data, \
    get_true_labels, aggregate_private_teacher_votes, aggregate_public_teachers_GNMax


K = 11
dataname = "fashion_mnist"
print("Data: {}".format(dataname))
if dataname == "mnist":
    epsilon_teacher = 0.0892
elif dataname == "fashion_mnist":
    epsilon_teacher = 0.0852
else:
    raise Exception("Unsupported dataset {}!".format(dataname))

plot = False

delta_teacher = 0.0001
m = 7

aggregated_epsilon = m * epsilon_teacher
aggregated_delta = m * delta_teacher

### compute lambda and sigma^2 for PATE (i.e., GNMax, Gaussian noise parameters)
lbd_min = (np.log(1 / aggregated_delta)) / aggregated_epsilon + 1 + 3
print('lbd min: {:.4f}'.format(lbd_min))
lbd_candidates = np.arange(lbd_min, 500)
sigma_sq_candidates = lbd_candidates / (aggregated_epsilon - np.log(1 / aggregated_delta) / (lbd_candidates - 1))
min_sigma_sq_idx = np.argmin(sigma_sq_candidates)
lbd = sigma_sq_candidates[min_sigma_sq_idx]
sigma_sq = sigma_sq_candidates[min_sigma_sq_idx]
sigma = np.sqrt(sigma_sq)
print('min sigma sq lambda: {:.4f}, min sigma sq: {:.4f}, min sigma: {:.4f}'.format(lbd, sigma_sq, sigma))
if plot:
    fontsize = 20
    plt.plot(lbd_candidates, sigma_sq_candidates)
    plt.xlabel("$\lambda$", fontsize=fontsize)
    plt.ylabel("$\sigma^2$", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    dataname_str = "MNIST" if dataname == "mnist" else "Fashion-MNIST"
    plt.title("Dataset: {}".format(dataname_str), fontsize=fontsize)
    # plt.show()
    plt.savefig("lbd_sigma_{}.pdf".format(dataname), bbox_inches="tight", pad_inches=0.1)
    plt.close()
print('eps teacher: {:.4f}, m: {:.4f}, aggregated eps: {:.4f}, delta: {:.4f}, aggregated delta: {:.4f}'
      .format(epsilon_teacher, m, aggregated_epsilon, delta_teacher, aggregated_delta))
print('lbd: {:.4f}, sigma^2: {:.4f}'.format(lbd, sigma_sq))

print('============')
num_query = 100
delta_prime = 0.0001

total_delta = num_query * aggregated_delta + delta_prime
total_eps = np.sqrt(2 * num_query * np.log(1/delta_prime)) * aggregated_epsilon \
            + num_query * aggregated_epsilon * (np.exp(aggregated_epsilon) - 1)

print("# Query {}, total privacy loss eps: {:.2f}, delta: {:.3f}".format(num_query, total_eps, total_delta))

device = "cpu"

# raise Exception("stop!")

### now proceed to compare PATE vs. aggregating private teachers
# load MNIST test data
# load test data

if dataname == "mnist":
    test_data = datasets.MNIST(
        root='data',
        train=False,
        transform=ToTensor()
    )
elif dataname == "fashion_mnist":
    test_data = datasets.FashionMNIST(
        root='data',
        train=False,
        transform=ToTensor()
    )
else:
    raise Exception("Unsupported dataset {}!".format(dataname))

# get test data from target classes
idx_test = (test_data.targets == 5) | (test_data.targets == 8)
test_data.data = test_data.data[idx_test]
test_data.targets = test_data.targets[idx_test]
print(test_data.data.size(), test_data.targets.size())  # torch.Size([1866, 28, 28]) torch.Size([1866])


def load_saved_teacher_models(save_folder, device="cpu"):
    list_teachers = []
    for teacher_idx in range(K):
        save_path = os.path.join(save_folder, 'teacher_{}.model'.format(teacher_idx))
        teacher_model = CNN()
        teacher_model.load_state_dict(torch.load(save_path, map_location=device))
        list_teachers.append(teacher_model)
    return list_teachers

def test_teachers_prediction(aggregated_votes, true_labels):
    acc = len(np.where(aggregated_votes == true_labels)[0]) / len(aggregated_votes)
    return acc


def one_exp_run(n_to_label):
    dataloader = prepare_dataloader_to_label(test_data, n_to_label=n_to_label, random=True)

    true_labels = get_true_labels(dataloader)
    print('true labels: ', true_labels)

    public_teachers_folder = "public_{}_teachers_test".format(dataname)
    list_public_teachers = load_saved_teacher_models(public_teachers_folder, device)
    public_teacher_responses = ask_teachers_to_label_data(list_public_teachers, dataloader, device)
    PATE_aggregated_votes = aggregate_public_teachers_GNMax(public_teacher_responses, sigma)
    PATE_acc = test_teachers_prediction(PATE_aggregated_votes, true_labels)
    print('PATE acc: {:.4f}'.format(PATE_acc))

    if dataname == "mnist":
        private_teachers_folder = "private_mnist_teachers_test_2"
    else:
        private_teachers_folder = "private_fashion_mnist_teachers_test"

    list_private_teachers = load_saved_teacher_models(private_teachers_folder, device)
    priv_teacher_responses = ask_teachers_to_label_data(list_private_teachers, dataloader, device)
    priv_teacher_aggregated_votes = aggregate_private_teacher_votes(priv_teacher_responses, m, epsilon_teacher,
                                               delta_0=delta_teacher, delta=aggregated_delta, aggregation_method='optimized')
    priv_teacher_acc = test_teachers_prediction(priv_teacher_aggregated_votes, true_labels)
    print('DaRRM (optimized gamma) acc: {:.4f}'.format(priv_teacher_acc))

    priv_teacher_responses_subsampling = aggregate_private_teacher_votes(priv_teacher_responses, m, epsilon_teacher,
                                               delta_0=delta_teacher, delta=aggregated_delta, aggregation_method='subsample')
    priv_teacher_acc_subsampling = test_teachers_prediction(priv_teacher_responses_subsampling, true_labels)
    print('DaRRM (subsampling gamma) acc: {:.4f}'.format(priv_teacher_acc_subsampling))

    return PATE_acc, priv_teacher_acc, priv_teacher_acc_subsampling


num_exp = 10

all_accs = {'pate': [], 'optimized': [], 'subsampling': []}
for exp_idx in range(num_exp):
    PATE_acc, priv_teacher_acc, priv_teacher_acc_subsampling = one_exp_run(num_query)
    all_accs['pate'].append(PATE_acc)
    all_accs['optimized'].append(priv_teacher_acc)
    all_accs['subsampling'].append(priv_teacher_acc_subsampling)

def compute_mean_std(all_accs, algo):
    data = np.array(all_accs[algo])
    print(algo, data)
    mean, std = np.mean(data), np.std(data)
    print('[{}], {:.2f} ({:.2f})'.format(algo, mean, std))

compute_mean_std(all_accs, 'pate')
compute_mean_std(all_accs, 'optimized')
compute_mean_std(all_accs, 'subsampling')
