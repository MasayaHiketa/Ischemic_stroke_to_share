import numpy as np
from abc import ABC
from sklearn.neighbors import KDTree
import scipy.stats as stats
from scipy.special import gamma
import multiprocessing as mp

class Gaussian:
    def __init__(self, mean, sigma):
        self.mean, self.sigma = mean, sigma
        self.dim = len(mean) if mean.ndim > 1 else 1
    def get_density(self, sample):
        return np.exp(-0.5*np.sum((np.power((sample-self.mean)/self.sigma, 2.))))

class VonMises:
    def __init__(self, mean, kappa):
        self.mean = mean
        self.kappa = kappa
    def get_density(self, sample):
        return stats.vonmises.pdf(self.mean, self.kappa, sample)

class KDE(ABC):
    def __init__(self, samples, k, beta):
        self.init_parameters(samples, k, beta)
        c = (1/np.sqrt(2*np.pi))**self.dim
        sigmas = self.compute_sigmas(samples)
        self.kernels = np.array([Gaussian(sample, sigma) for sample,sigma in zip(samples, sigmas)])
        self.weights = c/np.power(sigmas, self.dim)
    def compute_sigmas(self, samples):
        pass
    def init_parameters(self, samples, k, beta):
        samples = samples.reshape(-1, 1) if samples.ndim == 1 else samples
        self.k = k
        self.beta = beta
        self.dim = len(samples[0])
    def get_density(self, sample):
        densities = np.array([kernel.get_density(sample) for kernel in self.kernels])
        return np.mean(self.weights * densities)
    def get_densities(self, samples):
        return np.array([self.get_density(sample) for sample in samples])
    def get_densities_batch(self, samples, batch):
        """
            use case: avoid OOM
        """
        ret = np.empty(len(samples))
        for i in range(0, len(samples), batch):
            end_index = min(len(samples), i + batch)
            ret[i : end_index] = self.get_densities(samples[i:end_index])
        return ret
    def get_densities_mp(self, samples, process_num = 4, batch=None):
        if len(samples) < 1000:
            return self.get_densities(samples)
        indices = np.linspace(0, len(samples), process_num + 1).astype(int)
        ret_dict = mp.Manager().dict()
        processes = []
        for i in range(process_num):
            processes.append(
                mp.Process(
                    target=self.get_densities_mp_target,
                    args=(i, samples[indices[i]:indices[i+1]],batch,
                          ret_dict)
                )
            )
            processes[i].start()
        for process in processes:
            process.join()
        ret = np.empty(len(samples))
        for i in range(process_num):
            ret[indices[i]:indices[i+1]] = ret_dict[i]
        return ret
    def get_densities_mp_target(self, subprocess_id, samples, batch, ret_dict):
        if batch:
            ret_dict[subprocess_id] = self.get_densities_batch(samples, batch)
        else:
            ret_dict[subprocess_id] = self.get_densities(samples)
    def MSE(self, samples, true_densities):
        densities = self.get_densities(samples)
        return np.mean((densities-true_densities)**2)

class RVKDE(KDE):
    def init_parameters(self, samples, k, beta):
        super().init_parameters(samples, k, beta) 
        self.nn = KDTree(samples)
    def compute_sigmas(self, samples):
        R_scale = np.sqrt(np.pi)/np.power((self.k+1)*gamma(self.dim/2+1), 1./self.dim)
        sigma_scale = R_scale*self.beta*(self.dim+1.)/self.dim/self.k
        total_distances = self.compute_distances(samples)
        return total_distances * sigma_scale
    def query_neighbors(self, samples):
        samples = samples.reshape(-1, 1) if samples.ndim == 1 else samples
        return self.nn.query(samples,k=self.k+1)
    def compute_distances(self, samples, batch=10):
        # ret = np.empty(len(samples))
        # for i in range(0, len(samples), batch):
        #     end_index = min(len(samples), i + batch)
        #     distances, _ = self.query_neighbors(samples[i:end_index])
        #     ret[i:end_index] = np.sum(distances, axis=1)
        # return ret
        return np.array([np.sum(self.query_neighbors(np.array([sample]))[0]) for sample in samples])
        # distances, _ = self.query_neighbors(samples)
        # return np.sum(distances, axis = 1)

class ERAKDE(RVKDE):
    def __init__(self, samples, k, beta, factor):
        self.init_parameters(samples, k, beta, factor)
        sigmas = self.compute_erakde_sigmas(samples)
        c = (1/np.sqrt(2*np.pi))**self.dim
        self.kernels = np.array([Gaussian(sample, sigma) for sample, sigma in zip(samples, sigmas)])
        self.weights = c/np.power(sigmas, self.dim)
    def init_parameters(self, samples, k, beta, factor):
        super().init_parameters(samples, k, beta)
        self.factor = factor
    def compute_erakde_sigmas(self, samples):
        rvkde_sigmas = self.compute_rvkde_sigmas(samples)
        sd = self.compute_standard_distance(samples)
        diff = self.compute_elevation(sd, rvkde_sigmas)
        elevated_sigmas = rvkde_sigmas + diff if diff > 0 else rvkde_sigmas
        return elevated_sigmas
    def compute_rvkde_sigmas(self, samples):
        return self.compute_sigmas(samples)
    def compute_standard_distance(self, samples):
        return np.sqrt(np.sum(np.var(samples, axis=0)))
    def compute_elevation(self, sd, sigmas):
        return self.factor*((4/(len(sigmas)*(self.dim+2)))**(1./(self.dim+4)))*sd - np.median(sigmas)

class CircularERAKDE(ERAKDE):
    def __init__(self, samples, k, beta, factor, sample_max, sample_min, radius):
        self.init_parameters(k, beta, factor, sample_max, sample_min, radius)
        normalized_samples = self.normalize(samples.flatten())
        samples_2d = np.stack((np.cos(2*np.pi*normalized_samples),np.sin(2*np.pi*normalized_samples)),axis=1)
        self.nn = KDTree(samples_2d)
        self.elevated_sigmas = self.compute_elevated_sigmas(normalized_samples, samples_2d)
        sigmas = self.elevated_sigmas
        kappas = 1/sigmas**2
        self.kernels = np.array([VonMises(2*np.pi*sample, kappa) for sample, kappa in zip(normalized_samples, kappas)])
        self.weights = [1]*len(samples)
    def init_parameters(self, k, beta, factor, sample_max, sample_min, radius):
        """
            will transform range of samples from [sample_min, sample_max]
            to [0, self.sample_range]
        """
        self.k, self.beta, self.dim = k, beta, 1
        self.sample_range = sample_max - sample_min
        self.sample_min = sample_min
        self.radius = radius
        self.factor = factor
    def compute_elevated_sigmas(self, normalized_samples, samples_2d):
        sd = 2*np.pi*self.radius*self.compute_standard_distance(normalized_samples)
        print(sd)
        rvkde_sigmas = self.compute_rvkde_sigmas(samples_2d)
        print(rvkde_sigmas[:5])
        diff = self.compute_elevation(sd, rvkde_sigmas)
        print(diff)
        elevated_sigmas = rvkde_sigmas + diff if diff > 0 else rvkde_sigmas
        print(elevated_sigmas[:5])
        print((1/elevated_sigmas**2)[:5])
        return elevated_sigmas
    def compute_distances(self, samples, batch=100):
        # ret = np.empty(len(samples))
        # for i in range(0, len(samples), batch):
        #     end_index = min(len(samples), i + batch)
        #     distances, _ = self.query_neighbors(samples[i:end_index])
        #     ret[i:end_index] = self.radius*np.sum(np.arccos(1-0.5*distances**2), axis = 1)
        # return ret
        return np.array([self.radius*np.sum(np.arccos(1-0.5*self.query_neighbors(np.array([sample]))[0]**2)) for sample in samples])
    def normalize(self, samples):
        return (samples-self.sample_min)/self.sample_range
    def map_to_radians(self, samples):
        return 2*np.pi*self.normalize(samples)
    def get_density(self, sample):
        sample = self.map_to_radians(sample)
        return super().get_density(sample)
    def get_means(self):
        return np.array([kernel.mean for kernel in self.kernels])
    def get_kappas(self):
        return np.array([kernel.kappa for kernel in self.kernels])

class JitERAKDE(KDE):
    def __init__(self, samples, k, beta, factor):
        self.erakde = ERAKDE(samples, k, beta, factor)
        self.means = np.array([kernel.mean for kernel in self.erakde.kernels])
        self.sigmas = np.array([kernel.sigma for kernel in self.erakde.kernels])
        self.coefficients = self.erakde.weights
        self.dim = self.erakde.dim
        self.beta = self.erakde.beta
        self.k = self.erakde.k
        self.factor = self.erakde.factor
    def get_densities(self, samples):
        return self.get_densities_jit(samples, self.means, self.sigmas, self.coefficients)
    @staticmethod
    def get_densities_jit(samples, means, sigmas, coefficients):
        # almost equal owing to some order of computation is changed
        # return np.mean(coefficients * np.exp(-0.5 * (np.sum(((samples[:,np.newaxis] - means)**2), axis = 2) / sigmas ** 2)), axis=1)
        return np.array([np.mean(coefficients*np.exp(-0.5*np.sum(((sample-means)/sigmas[:,np.newaxis])**2,axis=1))) for sample in samples])

class JitCircularERAKDE(KDE):
    def __init__(self, samples, k, beta, factor, sample_max, sample_min, radius):
        self.circular_erakde = CircularERAKDE(samples, k, beta, factor, sample_max, sample_min, radius)
        self.means = self.circular_erakde.get_means()
        self.kappas = self.circular_erakde.get_kappas()
        self.coefficients = 1/(2*np.pi*np.i0(self.kappas))
    def get_densities(self,samples):
        samples = self.circular_erakde.map_to_radians(samples)
        return self.get_densities_jit(samples, self.means, self.kappas, self.coefficients)
    @staticmethod
    def get_densities_jit(samples, means, kappas, coefficients):
        # return np.mean(coefficients * np.exp(kappas*np.cos(samples[:,np.newaxis] - means)), axis=1)
        return np.array([np.mean(coefficients*np.exp(kappas*np.cos(sample - means))) for sample in samples])

class STERAKDE(KDE):
    def __init__(self,
                spatial_k, temporal_k,
                spatial_beta, temporal_beta,
                spatial_factor, temporal_factor,
                sample_max, sample_min,
                radius):
        # self.dim = 3
        self.spatial_k = spatial_k
        self.temporal_k = temporal_k
        self.spatial_beta = spatial_beta
        self.temporal_beta = temporal_beta
        self.spatial_factor = spatial_factor
        self.temporal_factor = temporal_factor
        self.sample_max = sample_max
        self.sample_min = sample_min
        self.radius = radius
    def fit(self, samples):
        self.dim = len(samples[0])
        # self.erakde = ERAKDE(samples[:,:2], self.spatial_k, self.spatial_beta, self.spatial_factor)
        self.erakde = ERAKDE(samples[:,:-1], self.spatial_k, self.spatial_beta, self.spatial_factor)
        self.circular_erakde = CircularERAKDE(samples[:,-1], self.temporal_k, self.temporal_beta, self.temporal_factor, self.sample_max, self.sample_min, self.radius)
        self.spatial_kernels = self.erakde.kernels
        self.temporal_kernels = self.circular_erakde.kernels
        self.spatial_weights = self.erakde.weights
        self.temporal_weights = self.circular_erakde.weights
        return self
    def get_density(self, sample):
        # spatial_densities = self.spatial_weights * np.array([kernel.get_density(sample[:2]) for kernel in self.spatial_kernels])
        spatial_densities = self.spatial_weights * np.array([kernel.get_density(sample[:-1]) for kernel in self.spatial_kernels])
        normalized_t_sample = self.circular_erakde.map_to_radians(sample[-1])
        temporal_densities = self.temporal_weights * np.array([kernel.get_density(normalized_t_sample) for kernel in self.temporal_kernels])
        return np.mean(spatial_densities * temporal_densities)
    def compute_sigmas(self, samples):
        pass

class JitSTERAKDE(STERAKDE):
    def __init__(self,
                spatial_k, temporal_k,
                spatial_beta, temporal_beta,
                spatial_factor, temporal_factor,
                sample_max, sample_min,
                radius):
        super().__init__(spatial_k, temporal_k,
                         spatial_beta, temporal_beta,
                         spatial_factor, temporal_factor,
                         sample_max, sample_min,
                         radius)
        self.sample_range = self.sample_max - self.sample_min
    def fit(self, samples):
        super().fit(samples)
        self.spatial_means = np.array([kernel.mean for kernel in self.spatial_kernels])
        self.temporal_means = np.array([kernel.mean for kernel in self.temporal_kernels])     
        self.sigmas = np.array([kernel.sigma for kernel in self.spatial_kernels])
        self.kappas = np.array([kernel.kappa for kernel in self.temporal_kernels])
        self.temporal_coefficients = 1/(2*np.pi*np.i0(self.kappas))
        del self.erakde
        del self.circular_erakde
        del self.spatial_kernels
        del self.temporal_kernels
        return self
    def read_sigmas(self, filename, samples):
        with open(filename, "r") as fp:
            lines = fp.readlines()
            self.sigmas = np.asarray(lines[0].strip().split(" "), dtype = float)
            temporal_sigmas = np.asarray(lines[1].strip().split(" "), dtype = float)
            self.kappas = 1/temporal_sigmas**2
        self.spatial_means = samples[:,:2]
        self.temporal_means = self.map_to_radians(samples[:,-1].flatten())
        self.spatial_weights = 1/(2*np.pi*self.sigmas**2)
        self.temporal_coefficients = 1/(2*np.pi*np.i0(self.kappas))
    def read_sigmas_kappas(self, filename, samples):
        with open(filename, "r") as fp:
            lines = fp.readlines()
            self.sigmas = np.asarray(lines[0].strip().split(" "), dtype = float)
            self.kappas = np.asarray(lines[1].strip().split(" "), dtype = float)
        self.spatial_means = samples[:,:2]
        self.temporal_means = self.map_to_radians(samples[:,-1].flatten())
        self.spatial_weights = 1/(2*np.pi*self.sigmas**2)
        self.temporal_coefficients = 1/(2*np.pi*np.i0(self.kappas))
    def normalize(self, samples):
        return (samples-self.sample_min)/self.sample_range
    def map_to_radians(self, samples):
        return 2*np.pi*self.normalize(samples)
    def get_densities_mp(self, samples, proccess_num = 4):
        temporal_samples = self.map_to_radians(samples[:,-1])
        adjusted_samples = np.concatenate((samples[:,:-1], temporal_samples.reshape(-1,1)),axis=1)
        if len(samples) < 1000:
            return self.get_densities_normalized(adjusted_samples)
        return super().get_densities_mp(adjusted_samples, proccess_num, None)
    def get_densities_mp_target(self, subprocess_id, samples, batch, ret_dict):
        ret_dict[subprocess_id] = np.array([self.get_density_normalized(sample) for sample in samples])
    def get_densities(self, samples):
        temporal_samples = self.map_to_radians(samples[:,-1])
        return self.get_densities_jit(samples[:,:-1], temporal_samples,
                                      self.spatial_means, self.sigmas,
                                      self.temporal_means, self.kappas,
                                      self.spatial_weights, self.temporal_coefficients)
    def get_density(self, sample):
        temporal_samples = self.map_to_radians(sample[-1])
        return self.get_density_jit(sample[:-1], temporal_samples,
                                    self.spatial_means, self.sigmas,
                                    self.temporal_means, self.kappas,
                                    self.spatial_weights, self.temporal_coefficients)
    def get_density_normalized(self, sample):
        return self.get_density_jit(sample[:-1], sample[-1],
                                    self.spatial_means, self.sigmas,
                                    self.temporal_means, self.kappas,
                                    self.spatial_weights, self.temporal_coefficients)
    def get_densities_normalized(self, samples):
        return self.get_densities_jit(samples[:,:-1], samples[:,-1],
                                      self.spatial_means, self.sigmas,
                                      self.temporal_means, self.kappas,
                                      self.spatial_weights, self.temporal_coefficients)
    @staticmethod
    def get_densities_jit(spatial_samples, temporal_samples,
                          spatial_means, sigmas,
                          temporal_means, kappas,
                          spatial_coefficients, temporal_coefficients):
        spatial_densities = np.array([spatial_coefficients*np.exp(-0.5*np.sum(((sample-spatial_means)/sigmas[:,np.newaxis])**2, axis=1)) for sample in spatial_samples])
        temporal_densities = np.array([temporal_coefficients * np.exp(kappas*np.cos(temporal_sample - temporal_means)) for temporal_sample in temporal_samples])
        # spatial_densities = spatial_coefficients * np.exp(-0.5 * (np.sum(((spatial_samples[:,np.newaxis] - spatial_means)**2), axis = 2) / sigmas ** 2))
        # temporal_densities = temporal_coefficients * np.exp(kappas*np.cos(temporal_samples[:,np.newaxis] - temporal_means))
        return np.mean(spatial_densities * temporal_densities, axis = 1)
    @staticmethod
    def get_density_jit(spatial_sample, temporal_sample,
                        spatial_means, sigmas,
                        temporal_means, kappas,
                        spatial_coefficients, temporal_coefficients):
        spatial_density = spatial_coefficients*np.exp(-0.5 * np.sum(((spatial_sample-spatial_means) / sigmas[:,np.newaxis]) ** 2, axis=1))
        temporal_density = temporal_coefficients * np.exp(kappas*np.cos(temporal_sample - temporal_means))
        return np.mean(spatial_density * temporal_density)
