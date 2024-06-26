"""make variations of input image"""

import argparse, os, sys, glob
import PIL
import torch
import numpy as np
import torchvision
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from scripts.wavelet_color_fix import wavelet_reconstruction, adaptive_instance_normalization
from scripts.helper import DDNMGuidance

def space_timesteps(num_timesteps, section_counts):
	"""
	Create a list of timesteps to use from an original diffusion process,
	given the number of timesteps we want to take from equally-sized portions
	of the original process.
	For example, if there's 300 timesteps and the section counts are [10,15,20]
	then the first 100 timesteps are strided to be 10 timesteps, the second 100
	are strided to be 15 timesteps, and the final 100 are strided to be 20.
	If the stride is a string starting with "ddim", then the fixed striding
	from the DDIM paper is used, and only one section is allowed.
	:param num_timesteps: the number of diffusion steps in the original
						  process to divide up.
	:param section_counts: either a list of numbers, or a string containing
						   comma-separated numbers, indicating the step count
						   per section. As a special case, use "ddimN" where N
						   is a number of steps to use the striding from the
						   DDIM paper.
	:return: a set of diffusion steps from the original process to use.
	"""
	if isinstance(section_counts, str):
		if section_counts.startswith("ddim"):
			desired_count = int(section_counts[len("ddim"):])
			for i in range(1, num_timesteps):
				if len(range(0, num_timesteps, i)) == desired_count:
					return set(range(0, num_timesteps, i))
			raise ValueError(
				f"cannot create exactly {num_timesteps} steps with an integer stride"
			)
		section_counts = [int(x) for x in section_counts.split(",")]   #[250,]
	size_per = num_timesteps // len(section_counts)
	extra = num_timesteps % len(section_counts)
	start_idx = 0
	all_steps = []
	for i, section_count in enumerate(section_counts):
		size = size_per + (1 if i < extra else 0)
		if size < section_count:
			raise ValueError(
				f"cannot divide section of {size} steps into {section_count}"
			)
		if section_count <= 1:
			frac_stride = 1
		else:
			frac_stride = (size - 1) / (section_count - 1)
		cur_idx = 0.0
		taken_steps = []
		for _ in range(section_count):
			taken_steps.append(start_idx + round(cur_idx))
			cur_idx += frac_stride
		all_steps += taken_steps
		start_idx += size
	return set(all_steps)

def chunk(it, size):
	it = iter(it)
	return iter(lambda: tuple(islice(it, size)), ())

class ImageDataset(Dataset):
	def __init__(self, init_img_dir, outpath, origin_img_dir=None, transform=None, gpu_id=0, gpu_num=1):
		self.init_img_dir = init_img_dir
		self.outpath = outpath
		self.transform = transform if transform else transforms.ToTensor()
		self.guidance = DDNMGuidance()
		self.guidance_func = self.guidance.forward_func
		# Filter out non-image files and optionally perform subsampling
		self.img_list = [img for img in sorted(os.listdir(init_img_dir)) if img.endswith(('.png', '.jpg', '.jpeg'))]
		# Exclude already processed images
		# self.img_list = [img for img in self.img_list if not os.path.exists(os.path.join(outpath, img))]
		self.origin_img_dir = init_img_dir.replace('inputs', 'gts')
		self.lensless_img_dir = init_img_dir.replace('inputs', 'sim_captures')
		self.img_list = self.img_list[gpu_id::gpu_num]
		# self.lensless_img_dir = "/root/caixin/StableSR/data/flatnet/sim_captures"
		self.decoded_rgb_capture_sim_path = '/root/caixin/StableSR/data/flatnet_val/inputs_backup'
	def __len__(self):
		return len(self.img_list)

	def __getitem__(self, idx):
		img_name = self.img_list[idx]
		img_path = os.path.join(self.init_img_dir, img_name)
		image = load_img(img_path)[0]
		if self.transform:
			image = self.transform(image)
		image = image.clamp(-1, 1)
		org_img_path = os.path.join(self.origin_img_dir, img_name)
		org_image = load_img(org_img_path)[0]
		if self.transform:
			org_image = self.transform(org_image)
		org_image = org_image.clamp(-1, 1)
		
		decoded_sim = os.path.join(self.decoded_rgb_capture_sim_path, img_name)
		decoded_sim = load_img(decoded_sim)[0]
		if self.transform:
			decoded_sim = self.transform(decoded_sim)
		decoded_sim = decoded_sim.clamp(-1, 1)	
		lensless_image = self.guidance_func(org_image)[0]
		return image, image

def load_model_from_config(config, ckpt, verbose=False):
	print(f"Loading model from {ckpt}")
	pl_sd = torch.load(ckpt, map_location="cpu")
	if "global_step" in pl_sd:
		print(f"Global Step: {pl_sd['global_step']}")
	sd = pl_sd["state_dict"]
	model = instantiate_from_config(config.model)
	m, u = model.load_state_dict(sd, strict=False)
	if len(m) > 0 and verbose:
		print("missing keys:")
		print(m)
	if len(u) > 0 and verbose:
		print("unexpected keys:")
		print(u)

	model.cuda()
	model.eval()
	return model

def load_img(path):
	image = Image.open(path).convert("RGB")
	w, h = image.size
	# print(f"loaded input image of size ({w}, {h}) from {path}")
	w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
	image = image.resize((w, h), resample=PIL.Image.LANCZOS)
	image = np.array(image).astype(np.float32) / 255.0
	image = image[None].transpose(0, 3, 1, 2)
	image = torch.from_numpy(image)
	return 2.*image - 1.




def main():
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--init-img",
		type=str,
		nargs="?",
		help="path to the input image",
		default="data/flatnet2single/inputs",
	)
	parser.add_argument(
		"--outdir",
		type=str,
		nargs="?",
		help="dir to write results to",
		default="data/flatnet2single/outputs_ft",
	)

	parser.add_argument(
		"--C",
		type=int,
		default=4,
		help="latent channels",
	)
	parser.add_argument(
		"--f",
		type=int,
		default=8,
		help="downsampling factor, most often 8 or 16",
	)
	parser.add_argument(
		"--n_samples",
		type=int,
		default=6,
		help="how many samples to produce for each given prompt. A.k.a batch size",
	)
	parser.add_argument(
		"--config",
		type=str,
		default="configs/stableSRNew/v2-finetune_lensless_T_512.yaml",
		help="path to config which constructs model",
	)
	parser.add_argument(
		"--ckpt",
		type=str,
		default="models/ldm/stable-diffusion-v1/model.ckpt",
		help="path to checkpoint of model",
	)
	parser.add_argument(
		"--vqgan_ckpt",
		type=str,
		default="models/ldm/stable-diffusion-v1/epoch=000011.ckpt",
		help="path to checkpoint of VQGAN model",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="the seed (for reproducible sampling)",
	)
	parser.add_argument(
		"--precision",
		type=str,
		help="evaluate at this precision",
		choices=["full", "autocast"],
		default="autocast"
	)
	parser.add_argument(
		"--input_size",
		type=int,
		default=512,
		help="input size",
	)
	# parser.add_argument(
	# 	"--dec_w",
	# 	type=float,
	# 	default=0.5,
	# 	help="weight for combining VQGAN and Diffusion",
	# )
	parser.add_argument(
		"--colorfix_type",
		type=str,
		default="nofix",
		help="Color fix type to adjust the color of HR result according to LR input: adain (used in paper); wavelet; nofix",
	)

	parser.add_argument(
		"--ddim_steps",
		type=int,
		default=50,
		help="number of ddim sampling steps",
	)
	parser.add_argument(
		"--ddim_eta",
		type=float,
		default=0.0,
		help="ddim eta (eta=0.0 corresponds to deterministic sampling",
	)
	parser.add_argument(
		"--scale",
		type=float,
		default=7.0,
		help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
	)
	parser.add_argument(
		"--strength",
		type=float,
		default=0.75,
		help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
	)	

	parser.add_argument(
		"--use_negative_prompt",
		action='store_true',
		help="if enabled, save inputs",
	)

	opt = parser.parse_args()
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	print('>>>>>>>>>>color correction>>>>>>>>>>>')
	if opt.colorfix_type == 'adain':
		print('Use adain color correction')
	elif opt.colorfix_type == 'wavelet':
		print('Use wavelet color correction')
	else:
		print('No color correction')
	print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

	# vqgan_config = OmegaConf.load("configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml")
	# vq_model = load_model_from_config(vqgan_config, opt.vqgan_ckpt)
	# vq_model = vq_model.to(device)
	# vq_model.decoder.fusion_w = opt.dec_w

	seed_everything(opt.seed)

	transform = torchvision.transforms.Compose([
		torchvision.transforms.Resize(opt.input_size),
		torchvision.transforms.CenterCrop(opt.input_size),
	])

	config = OmegaConf.load(f"{opt.config}")
	model = load_model_from_config(config, f"{opt.ckpt}")
	model = model.to(device)

	sampler = DDIMSampler(model)
	ddnm_guidance = DDNMGuidance(opt)
	os.makedirs(opt.outdir, exist_ok=True)
	outpath = opt.outdir

	batch_size = opt.n_samples



	image_dataset = ImageDataset(opt.init_img, outpath, transform=transform)
	image_dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size // 2, pin_memory=True)

	model.register_schedule(given_betas=None, beta_schedule="linear", timesteps=1000,
						  linear_start=0.00085, linear_end=0.0120, cosine_s=8e-3)
	model.num_timesteps = 1000

	# sqrt_alphas_cumprod = copy.deepcopy(model.sqrt_alphas_cumprod)
	# sqrt_one_minus_alphas_cumprod = copy.deepcopy(model.sqrt_one_minus_alphas_cumprod)

	# use_timesteps = set(space_timesteps(1000, [opt.ddpm_steps]))
	# last_alpha_cumprod = 1.0
	# new_betas = []
	# timestep_map = []
	# for i, alpha_cumprod in enumerate(model.alphas_cumprod):
	# 	if i in use_timesteps:
	# 		new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
	# 		last_alpha_cumprod = alpha_cumprod
	# 		timestep_map.append(i)
	# new_betas = [beta.data.cpu().numpy() for beta in new_betas]
	# model.register_schedule(given_betas=np.array(new_betas), timesteps=len(new_betas))
	# model.num_timesteps = 1000
	# model.ori_timesteps = list(use_timesteps)
	# model.ori_timesteps.sort()
	model = model.to(device)

	ddim_timesteps = set(space_timesteps(1000, [opt.ddim_steps]))
	ddim_timesteps = list(ddim_timesteps)
	ddim_timesteps.sort()

	sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)


	# param_list = []
	# untrain_paramlist = []
	# name_list = []
	# for k, v in model.named_parameters():
	# 	if 'spade' in k or 'structcond_stage_model' in k:
	# 		param_list.append(v)
	# 	else:
	# 		name_list.append(k)
	# 		untrain_paramlist.append(v)
	# trainable_params = sum(p.numel() for p in param_list)
	# untrainable_params = sum(p.numel() for p in untrain_paramlist)
	# print(name_list)
	# print(trainable_params)
	# print(untrainable_params)

	# param_list = []
	# untrain_paramlist = []
	# for k, v in vq_model.named_parameters():
	# 	if 'fusion_layer' in k:
	# 		param_list.append(v)
	# 	elif 'loss' not in k:
	# 		untrain_paramlist.append(v)
	# trainable_params += sum(p.numel() for p in param_list)
	# # untrainable_params += sum(p.numel() for p in untrain_paramlist)
	# print(trainable_params)
	# print(untrainable_params)

	precision_scope = autocast if opt.precision == "autocast" else nullcontext
	niqe_list = []
	with torch.no_grad():
		with precision_scope("cuda"):

			with model.ema_scope():
				tic = time.time()
				count = 0
				for init_image, lensless_image in tqdm(image_dataloader):
					init_image = init_image.to(device)
					lensless_image = lensless_image.to(device)
					ddnm_guidance.y = lensless_image
					init_latent_generator = model.encode_first_stage(init_image)
					init_latent = model.get_first_stage_encoding(init_latent_generator)
					text_init = ['']*init_image.size(0)
					semantic_c = model.cond_stage_model(text_init)

					if opt.use_negative_prompt:
						negative_text_init = ['3d, cartoon, anime, sketches, (worst quality:2), (low quality:2)']*init_image.size(0)
						# negative_text_init = ['Bad photo.']*init_image.size(0)
						nega_semantic_c = model.cond_stage_model(negative_text_init)

					noise = torch.randn_like(init_latent)
					# If you would like to start from the intermediate steps, you can add noise to LR to the specific steps.
					t = repeat(torch.tensor([999]), '1 -> b', b=init_image.size(0))
					t = t.to(device).long()
					x_T = model.q_sample(x_start=init_latent, t=t, noise=noise)
					x_T = None

					samples, _ = sampler.ddim_sampling_sr_t(cond=semantic_c, struct_cond=init_latent, shape=init_latent.shape, ddnm_guidance=ddnm_guidance,
												unconditional_conditioning=nega_semantic_c if opt.use_negative_prompt else None,
												unconditional_guidance_scale=opt.scale if opt.use_negative_prompt else None,
												timesteps=np.array(ddim_timesteps),
												 x_T=x_T)
					x_samples = model.decode_first_stage(samples)
					if opt.colorfix_type == 'adain':
						x_samples = adaptive_instance_normalization(x_samples, init_image)
					elif opt.colorfix_type == 'wavelet':
						x_samples = wavelet_reconstruction(x_samples, init_image)
					x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
					# count += 1
					# if count==5:
					# 	tic = time.time()
					# if count >= 15:
					# 	print('>>>>>>>>>>>>>>>>>>>>>>>>')
					# 	print(time.time()-tic)
					# 	print(s)

					for i in range(init_image.size(0)):
						img_name = image_dataset.img_list.pop(0)
						basename = os.path.splitext(os.path.basename(img_name))[0]
						x_sample = 255. * rearrange(x_samples[i].cpu().numpy(), 'c h w -> h w c')
						Image.fromarray(x_sample.astype(np.uint8)).save(
							os.path.join(outpath, basename+'.png'))

				toc = time.time()

	print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
		  f" \nEnjoy.")


if __name__ == "__main__":
	main()
