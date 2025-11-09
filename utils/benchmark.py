import torch
import torch.nn as nn
from utils.trainer import model_init_
from utils.build import check_cfg, build_from_cfg
import os
import glob
from torchvision import transforms, datasets
from PIL import Image, ImageDraw, ImageFont
import time
from graphic.RawDataProcessor import SAMPLES_FREQUENCY, generate_images
import imageio
import sys
import cv2
import numpy as np
from torch.utils.data import DataLoader
import hashlib
import matplotlib.pyplot as plt
from threading import Thread, Event
from queue import Queue
from scipy.io import wavfile
from scipy.signal import stft, windows
from io import BytesIO

try:
    from DetModels import YOLOV5S
    from DetModels.yolo.basic import LoadImages, Profile, Path, non_max_suppression, Annotator, scale_boxes, colorstr, \
        Colors, letterbox

except ImportError:
    pass


# Current directory and metric directory
current_dir = os.path.dirname(os.path.abspath(__file__))
METRIC = os.path.join(current_dir, './metrics')

sys.path.append(METRIC)
sys.path.append(current_dir)
sys.path.append('utils/DetModels/yolo')

try:
    from .metrics.base_metric import EVAMetric
except ImportError:
    pass

from logger import colorful_logger


# Supported image and raw data extensions
image_ext = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
raw_data_ext = ['.iq', '.dat', '.wav']

def STFT(data,
         onside: bool = True,
         stft_point: int = 1024,
         fs: int = 100e6,
         duration_time: float = 0.1,
         ):

    """
    Performs Short-Time Fourier Transform (STFT) on the given data.

    Parameters:
    - data (array-like): Input data.
    - onside (bool): Whether to return one-sided or two-sided STFT, default is True.
    - stft_point (int): Number of points for STFT, default is 1024.
    - fs (int): Sampling frequency, default is 100 MHz.
    - duration_time (float): Duration time for each segment, default is 0.1 seconds.

    Returns:
    - f (array): Frequencies.
    - t (array): Times.
    - Zxx (array): STFT result.
    """

    slice_point = int(fs * duration_time)

    f, t, Zxx = stft(data[0: slice_point], fs,
         return_onesided=onside, window=windows.hamming(stft_point), nperseg=stft_point)

    return f, t, Zxx

def visualize_frames(frame_queue, stop_event):
    """
    Continuously display frames from the queue in a Matplotlib window.
    """
    plt.ion()  # interactive mode on
    fig, ax = plt.subplots()
    im = None

    while not stop_event.is_set():
        frame = frame_queue.get()
        if frame is None:  # sentinel value to exit
            break

        if im is None:
            im = ax.imshow(frame, aspect='auto')
        else:
            im.set_data(frame)

        ax.axis('off')
        plt.pause(0.01)  # allow GUI to update
    plt.ioff()
    plt.close(fig)

class Classify_Model(nn.Module):
    """
    A class representing a classification model for performing inference and benchmarking using a pre-trained model.

    Attributes:
    - logger (colorful_logger): Logger for logging messages with color.
    - cfg (str): Path to configuration dictionary.
    - device (str): Device to use for inference (CPU or GPU).
    - model (torch.nn.Module): Pre-trained model.
    - save_path (str): Path to save the results.
    - save (bool): Flag to indicate whether to save the results.
    """

    def __init__(self,
                 cfg: str = '../configs/exp1_test.yaml',
                 weight_path: str = '../default.path',
                 save: bool = True,
                 ):

        """
        Initializes the Classify_Model.

        Parameters:
        - cfg (str): Path to configuration dictionary.
        - weight_path (str): Path to the pre-trained model weights.
        - save (bool): Flag to indicate whether to save the results.
        """

        super().__init__()
        self.logger = self.set_logger

        if check_cfg(cfg):
            self.logger.log_with_color(f"Using config file: {cfg}")
            self.cfg = build_from_cfg(cfg)

        if self.cfg['device'] == 'cuda':
            if torch.cuda.is_available():
                self.logger.log_with_color("Using GPU for inference")
                self.device = self.cfg['device']
        else:
            self.logger.log_with_color("Using CPU for inference")
            self.device = "cpu"

        if os.path.exists(weight_path):
            self.logger.log_with_color(f"Using weight file: {weight_path}")
            self.weight_path = weight_path
        else:
            raise FileNotFoundError(f"weight path: {weight_path} does not exist")

        self.model = self.load_model
        self.model.to(self.device)
        self.model.eval()
        self.save_path = None
        self.cache_path = "./cache"

        self.save = save

        self.frame_queue = Queue(maxsize=2)  # small queue to avoid lag

    def start_visualization(self):
        self.stop_event = Event()
        self.vis_thread = Thread(target=visualize_frames, args=(self.frame_queue,self.stop_event))
        self.vis_thread.start()

    def terminate_visualization(self):
        self.frame_queue.put(None)
        self.vis_thread.join()
        self.frame_queue = None

    def inference(self, source='../example/', save_path: str = '../result'):
        """
        Performs inference on the given source data.

        Parameters:
        - source (str): Path to the source data.
        - save_path (str): Path to save the results.
        """
        torch.no_grad()
        if self.save:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            self.save_path = save_path
            self.logger.log_with_color(f"Saving results to: {save_path}")

        if not os.path.exists(source):
            self.logger.log_with_color(f"Source {source} dose not exit")

        # dir detect
        if os.path.isdir(source):
            data_list = glob.glob(os.path.join(source, '*'))

            for data in data_list:
                # detect images in dir
                if is_valid_file(data, image_ext):
                    self.ImgProcessor(data)
                # detect raw datas in dir
                elif is_valid_file(data, raw_data_ext):
                    self.RawdataProcess(data)
                else:
                    continue

        # detect single image
        elif is_valid_file(source, image_ext):
            self.ImgProcessor(source)

        # detect single pack of raw data
        elif is_valid_file(source, raw_data_ext):
            self.RawdataProcess(source)

    def optimizedGpuInference(self, source='../example/', save_path: str = '../result'):
        """
        Performs inference on the given source data.

        Parameters:
        - source (str): Path to the source data.
        - save_path (str): Path to save the results.
        """
        torch.no_grad()
        if self.save:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            self.save_path = save_path
            self.logger.log_with_color(f"Saving results to: {save_path}")

        if not os.path.exists(source):
            self.logger.log_with_color(f"Source {source} dose not exit")

        if not is_valid_file(source, raw_data_ext):
            self.logger.log_with_color(f"Source {source} is not valid raw data")

        """
        Transforming raw data into a video and performing inference on video.

        Parameters:
        - source (str): Path to the raw data.
        """
        # "iq" is a series of two values .. of float32
        samples_count = os.path.getsize(source) / 2 / 4
        duration = samples_count / SAMPLES_FREQUENCY
        print("PROCESSING", source, "(", duration, "s)")

        sample_duration_s = 0.1
        fs = 20e6
        stft_point = 1024

        if source.endswith('.wav'):
            # The returned 'data' is composed of tuples of float32 values.
            _sampling_rate, data = wavfile.read(source)
            fs = _sampling_rate
            # This also works:
            # np.complex64 means both the real and imaginary parts are stored as 32-bit (single-precision) floating-point numbers.
            # data1 = data.flatten().view(np.complex64)
            I = data[:, 0]
            Q = data[:, 1]
            data = I + 1j * Q
        else:
            # Load an array of int16
            data = np.fromfile(source, dtype=np.int16)
            # Pair the values in two, as complex numbers
            data = data[::2] + data[1::2] * 1j

        slice_point = int(fs * sample_duration_s)

        i = 0
        name = os.path.splitext(os.path.basename(source))[0]

        cmap = plt.cm.get_cmap("jet", 256)
        cmap_np = cmap(range(256))[:, :3]  # ignore alpha
        gpu_cmap = torch.tensor(cmap_np, dtype=torch.float32).cuda()  # (256,3)

        res = []
        while (i + 1) * slice_point <= len(data):
            start = int(i * slice_point)
            end = int((i + 1) * slice_point)
            time = i * sample_duration_s
            i += 0.5



            x = torch.from_numpy(data[start:end]).float().cuda()  # move to GPU

            n_fft = stft_point
            hop_length = n_fft  # or choose overlap

            # Create window
            window = torch.hamming_window(n_fft, device='cuda')

            # 1D STFT using torch.stft
            Zxx = torch.stft(x, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True, center=False)  # shape: (freq_bins, time_frames), complex64

            # Shift frequency axis (like np.fft.fftshift)
            Zxx = torch.roll(Zxx, shifts=Zxx.shape[0] // 2, dims=0)

            # Log magnitude (dB)
            amplitudes = 10 * torch.log10(torch.abs(Zxx) + 1e-12).unsqueeze(0)  # shape: (1, freq_bins, time_frames)
            transform = transforms.Compose([
                transforms.Resize((1440, 1920)),
            ])
            amplitudes = transform(amplitudes)[0]  # shape (H, W)

            #  Normalize to 0..1
            aug_min = amplitudes.min()
            aug_max = amplitudes.max()
            norm_amplitudes = (amplitudes - aug_min) / (aug_max - aug_min + 1e-9)  # avoids division by zero

            # Scale to 0..255 and convert to long indices
            indices = (norm_amplitudes * 255).long().clamp(0, 255)

            gpu_image = gpu_cmap[indices]   # shape (H, W, 3)

            transform = transforms.Compose([
                transforms.Resize((self.cfg['image_size'], self.cfg['image_size'])),
            ])
            preprocessed_image = transform(gpu_image.permute(2, 0, 1)).unsqueeze(0) # shape (1, 3, H, W)

            probabilities = torch.softmax(self.model(preprocessed_image), dim=1)

            predicted_class_index = torch.argmax(probabilities, dim=1).item()
            if probabilities[0][predicted_class_index] > 0.7 :
                predicted_class_name = get_key_from_value(self.cfg['class_names'], predicted_class_index)
            else :
                predicted_class_name = "no detection"
            print(f"{predicted_class_name}, probabilities: {probabilities[0]} {time}")

            image=(gpu_image.detach().cpu().numpy() *255).astype("uint8")

            frame = self.add_result(res=predicted_class_name,
                                probability=probabilities[0][predicted_class_index].item() * 100,
                                image=Image.fromarray(image),
                                time=time,
                                duration=sample_duration_s)

            if not self.frame_queue.full():
                self.frame_queue.put(frame)

            res.append(frame)

        imageio.mimsave(os.path.join(self.save_path, name + '.mp4'), res, fps=5)


    def liveGpuInference(self, source='../example/'):

        """
        Transforming raw data into a video and performing inference on video.

        Parameters:
        - source (str): Path to the raw data.
        """
        # "iq" is a series of two values .. of float32
        samples_count = os.path.getsize(source) / 2 / 4
        duration = samples_count / SAMPLES_FREQUENCY
        print("PROCESSING", source, "(", duration, "s)")

        sample_duration_s = 0.1
        fs = 20e6
        stft_point = 1024

        if source.endswith('.wav'):
            # The returned 'data' is composed of tuples of float32 values.
            _sampling_rate, data = wavfile.read(source)
            fs = _sampling_rate
            # This also works:
            # np.complex64 means both the real and imaginary parts are stored as 32-bit (single-precision) floating-point numbers.
            # data1 = data.flatten().view(np.complex64)
            I = data[:, 0]
            Q = data[:, 1]
            data = I + 1j * Q
        else:
            # Load an array of int16
            data = np.fromfile(source, dtype=np.int16)
            # Pair the values in two, as complex numbers
            data = data[::2] + data[1::2] * 1j

        slice_point = int(fs * sample_duration_s)

        i = 0
        name = os.path.splitext(os.path.basename(source))[0]

        cmap = plt.cm.get_cmap("jet", 256)
        cmap_np = cmap(range(256))[:, :3]  # ignore alpha
        gpu_cmap = torch.tensor(cmap_np, dtype=torch.float32).cuda()  # (256,3)

        res = []
        while (i + 1) * slice_point <= len(data):
            start = int(i * slice_point)
            end = int((i + 1) * slice_point)
            f, t, Zxx = STFT(data[start:end],
                         stft_point=stft_point, fs=fs, duration_time=0.1, onside=False)
            f = np.fft.fftshift(f)
            Zxx = np.fft.fftshift(Zxx, axes=0)
            aug = 10 * np.log10(np.abs(Zxx))
            extent = [t.min(), t.max(), f.min(), f.max()]

            plt.figure()
            plt.imshow(aug, extent=extent, aspect='auto', origin='lower', cmap='jet')
            plt.axis('off')
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=None, hspace=None)

            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300)
            plt.close()

            buffer.seek(0)
            image = Image.open(buffer)

            time = i * sample_duration_s / 2
            temp = self.model(self.preprocess(image))

            i += 1

            probabilities = torch.softmax(temp, dim=1)

            predicted_class_index = torch.argmax(probabilities, dim=1).item()
            if probabilities[0][predicted_class_index] > 0.7 :
                predicted_class_name = get_key_from_value(self.cfg['class_names'], predicted_class_index)
            else :
                predicted_class_name = "no detection"
            print("{}, probabilities: {} {} {}", predicted_class_name or "no detection", probabilities[0], time, image)

            frame = self.add_result(res=predicted_class_name,
                                probability=probabilities[0][predicted_class_index].item() * 100,
                                image=image,
                                time=time,
                                duration=sample_duration_s)

            if not self.frame_queue.full():
                self.frame_queue.put(frame)


    def forward(self, img):

        """
        Forward pass through the model.

        Parameters:
        - img (torch.Tensor): Input image tensor.

        Returns:
        - probability (float): Confidence probability of the predicted class.
        - predicted_class_name (str): Name of the predicted class.
        """

        self.model.eval()
        temp = self.model(img)
        probabilities = torch.softmax(temp, dim=1)
        predicted_class_index = torch.argmax(probabilities, dim=1).item()
        predicted_class_name = get_key_from_value(self.cfg['class_names'], predicted_class_index)
        probability = probabilities[0][predicted_class_index].item() * 100
        return probability, predicted_class_name

    @property
    def load_model(self):
        """
        Loads the pre-trained model.

        Returns:
        - model (torch.nn.Module): Loaded model.
        """

        self.logger.log_with_color(f"Using device: {self.device}")
        model = model_init_(self.cfg['model'], self.cfg['num_classes'], pretrained=True)

        if os.path.exists(self.weight_path):
            self.logger.log_with_color(f"Loading init weights from: {self.weight_path}")
            state_dict = torch.load(self.weight_path, map_location=self.device)
            model.load_state_dict(state_dict)
            self.logger.log_with_color(f"Successfully loaded pretrained weights from: {self.weight_path}")
        else:
            self.logger.log_with_color(f"init weights file not found at: {self.weight_path}. Skipping weight loading.")

        return model

    def ImgProcessor(self, source):
        """
         Performs inference on spectromgram data.

        Parameters:
        - source (str): Path to the image.
        """

        start_time = time.time()

        name = os.path.basename(source)[:-4]
        origin_image = Image.open(source).convert('RGB')
        preprocessed_image = self.preprocess(source)

        temp = self.model(preprocessed_image)

        probabilities = torch.softmax(temp, dim=1)

        predicted_class_index = torch.argmax(probabilities, dim=1).item()
        predicted_class_name = get_key_from_value(self.cfg['class_names'], predicted_class_index)

        end_time = time.time()
        self.logger.log_with_color(f"Inference time: {(end_time - start_time) / 100 :.8f} sec")
        self.logger.log_with_color(f"{source} contains Drone: {predicted_class_name}, "
                                   f"confidence1: {probabilities[0][predicted_class_index].item() * 100 :.2f} %,"
                                   f" start saving result")

        if self.save:
            res = self.add_result(res=predicted_class_name,
                                  probability=probabilities[0][predicted_class_index].item() * 100,
                                  image=origin_image)

            res.save(os.path.join(self.save_path, name + '.jpg'))

    def RawdataProcess(self, source):
        """
        Transforming raw data into a video and performing inference on video.

        Parameters:
        - source (str): Path to the raw data.
        """
        # "iq" is a series of two values .. of float32
        samples_count = os.path.getsize(source) / 2 / 4
        duration = samples_count / SAMPLES_FREQUENCY
        print("PROCESSING", source, "(", duration, "s)")
        location = os.path.join(self.cache_path, path_to_tmp(source, 'images_'))
        sample_duration_s = 0.1
        ratio = 1
        generate_images(source, duration_time=sample_duration_s, location=location, ratio=ratio, fs = 20e6)
        name = os.path.splitext(os.path.basename(source))[0]

        res = []

        for (i, image) in enumerate(sorted(os.listdir(location))):
            time = i * sample_duration_s / (2 ** ratio)
            image = Image.open(os.path.join(location, image))
            temp = self.model(self.preprocess(image))

            probabilities = torch.softmax(temp, dim=1)

            predicted_class_index = torch.argmax(probabilities, dim=1).item()
            if probabilities[0][predicted_class_index] > 0.7 :
                predicted_class_name = get_key_from_value(self.cfg['class_names'], predicted_class_index)
            else :
                predicted_class_name = "no detection"
            print("{}, probabilities: {} {} {}", predicted_class_name or "no detection", probabilities[0], time, image)

            _ = self.add_result(res=predicted_class_name,
                                probability=probabilities[0][predicted_class_index].item() * 100,
                                image=image,
                                time=time,
                                duration=sample_duration_s)
            res.append(_)

        imageio.mimsave(os.path.join(self.save_path, name + '.mp4'), res, fps=5)

    def add_result(self,
                   res,
                   image,
                   position=(40, 45),
                   font="DejaVuSans-Bold.ttf",
                   font_size=50,
                   text_color=(0, 0, 0),
                   probability=0.0,
                   time=None,
                   duration=None,
                   ):
        """
        Adds the inference result to the image.

        Parameters:
        - res (str): Inference result.
        - image (PIL.Image): Input image.
        - position (tuple): Position to add the text.
        - font (str): Font file path.
        - font_size (int): Font size.
        - text_color (tuple): Text color.
        - probability (float): Confidence probability.

        Returns:
        - image (PIL.Image): Image with added result.
        """
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype(font, font_size)
        except:
            font = ImageFont.load_default()
        minutes = int(time / 60)
        seconds_with_fraction = time % 60
        draw.text(position, f"{minutes:02d}:{seconds_with_fraction:05.2f}", fill=text_color, font=font)
        position = (position[0], position[1]*2)
        if res:
            draw.text(position, f"{res}", fill=text_color, font=font)
        else:
            draw.text(position, f"―", fill=text_color, font=font)
        return image

    @property
    def set_logger(self):
        """
        Sets up the logger.

        Returns:
        - logger (colorful_logger): Logger instance.
        """
        logger = colorful_logger('Inference')
        return logger

    def preprocess(self, img):

        transform = transforms.Compose([
            transforms.Resize((self.cfg['image_size'], self.cfg['image_size'])),
            transforms.ToTensor(),
        ])

        # TODO: Image.open(img) if it's a path
        image = img.convert('RGB')
        preprocessed_image = transform(image)

        preprocessed_image = preprocessed_image.to(self.device)
        preprocessed_image = preprocessed_image.unsqueeze(0)

        return preprocessed_image

    def benchmark(self, data_path, save_path=None):

        """
        Performs benchmarking on the given data and calculates evaluation metrics.

        Parameters:
        - data_path (str): Path to the benchmark data.

        Returns:
        - metrics (dict): Dictionary containing evaluation metrics.
        """
        snrs = os.listdir(data_path)

        if not save_path:
            save_path = os.path.join(data_path, 'benchmark result')
            if not os.path.exists(save_path): os.mkdir(save_path)

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        #根据得到映射关系写下面的，我得到的是★ 最佳映射 pred → gt: {0: 2, 1: 1, 2: 3, 3: 4, 4: 0}
        #MAP_P2G=torch.tensor([2,1,3,4,0],device=self.cfg['device'])
        #INV_MAP=torch.argsort(MAP_P2G)
        with torch.no_grad():
            for snr in snrs:
                CMS = os.listdir(os.path.join(data_path, snr))
                for CM in CMS:
                    stat_time = time.time()
                    self.model.eval()
                    _dataset = datasets.ImageFolder(
                        root=os.path.join(data_path, snr, CM),
                        transform=transforms.Compose([
                        transforms.Resize((self.cfg['image_size'], self.cfg['image_size'])),
                        transforms.ToTensor(),]),
                        allow_empty=True
                    )
                    dataset = DataLoader(_dataset, batch_size=self.cfg['batch_size'], shuffle=self.cfg['shuffle'])
                    print("Starting Benchmark...")

                    correct = 0
                    total = 0
                    probabilities = []
                    total_labels = []
                    classes_name = tuple(self.cfg['class_names'].keys())
                    cm_raw = np.zeros((5, 5), dtype=int)
                    for images, labels in dataset:
                        images, labels = images.to(self.cfg['device']), labels.to(self.cfg['device'])
                        outputs = self.model(images)
                        #outputs=outputs[:,INV_MAP]
                        #probs =torch.softmax(outputs,dim=1)
                        for output in outputs:
                            probabilities.append(list(torch.softmax(output, dim=0)))
                        _, predicted = outputs.max(1)
                        for p, t in zip(predicted.cpu(), labels.cpu()):
                            cm_raw[p,t]+=1
                        cm_raw[p, t] += 1   # 行 = pred, 列 = gt
                        total += labels.size(0)
                        correct += predicted.eq(labels).sum().item()
                        total_labels.append(labels)
                    _total_labels = torch.concat(total_labels, dim=0)
                    _probabilities = torch.tensor(probabilities)

                    metrics = EVAMetric(preds=_probabilities.to(self.cfg['device']),
                                        labels=_total_labels,
                                        num_classes=self.cfg['num_classes'],
                                        tasks=('f1', 'precision', 'CM'),
                                        topk=(1, 3, 5),
                                        save_path=save_path,
                                        classes_name=classes_name,
                                        pic_name=f'{snr}_{CM}')
                    metrics['acc'] = 100 * correct / total

                    s = (f'{snr} ' + f'CM: {CM} eva result:' + ' acc: ' + f'{metrics["acc"]}' + ' top-1: ' +
                         f'{metrics["Top-k"]["top1"]}' + ' top-1: ' + f'{metrics["Top-k"]["top1"]}' +
                         ' top-2 ' + f'{metrics["Top-k"]["top2"]}' + ' top-3 ' + f'{metrics["Top-k"]["top3"]}' +
                         ' mAP: ' + f'{metrics["mAP"]["mAP"]}' + ' macro_f1: ' + f'{metrics["f1"]["macro_f1"]}' +
                         ' micro_f1 : ' + f' {metrics["f1"]["micro_f1"]}\n')
                    txt_path = os.path.join(save_path, 'benchmark_result.txt')
                    colorful_logger(f'cost {(time.time()-stat_time)/60} mins')
                    with open(txt_path, 'a') as file:
                        file.write(s)

                print(f'{CM} Done!')
            print(f'{snr} Done!')
        row_ind, col_ind = linear_sum_assignment(-cm_raw)   # 取负→最大化对角线
        mapping_pred2gt = {int(r): int(c) for r, c in zip(row_ind, col_ind)}
        print("\n★ 最佳映射 pred → gt:", mapping_pred2gt)

        # 若要保存下来以后用：
        import json
        json.dump(mapping_pred2gt, open('class_to_idx_pred2gt.json', 'w'))
        print("映射已保存到 class_to_idx_pred2gt.json")

class Detection_Model:

    """
    A common interface for initializing and running different detection models.

    This class provides methods to initialize and run object detection models such as YOLOv5 and Faster R-CNN.
    It allows for easy switching between different models by providing a unified interface.

    Attributes:
    - S1model: The initialized detection model (e.g., YOLOv5S).
    - model_name: The name of the detection model to be used.
    - weight_path: The path to the pre-trained model weights.

    Methods:
    - __init__(self, cfg=None, model_name=None, weight_path=None):
        Initializes the detection model based on the provided configuration or parameters.
        If a configuration dictionary `cfg` is provided, it will be used to set the model name and weight path.
        Otherwise, the `model_name` and `weight_path` parameters can be specified directly.

    - yolov5_detect(self, source='../example/source/', save_dir='../res', imgsz=(640, 640), conf_thres=0.6, iou_thres=0.45, max_det=1000, line_thickness=3, hide_labels=True, hide_conf=False):
        Runs YOLOv5 object detection on the specified source.
        - source: Path to the input image or directory containing images.
        - save_dir: Directory to save the detection results.
        - imgsz: Image size for inference (height, width).
        - conf_thres: Confidence threshold for filtering detections.
        - iou_thres: IoU threshold for non-maximum suppression.
        - max_det: Maximum number of detections per image.
        - line_thickness: Thickness of the bounding box lines.
        - hide_labels: Whether to hide class labels in the output.
        - hide_conf: Whether to hide confidence scores in the output.

    - faster_rcnn_detect(self, source='../example/source/', save_dir='../res', weight_path='../example/detect/', imgsz=(640, 640), conf_thres=0.25, iou_thres=0.45, max_det=1000, line_thickness=3, hide_labels=False, hide_conf=False):
        Placeholder method for running Faster R-CNN object detection.
        This method is currently not implemented and should be replaced with the actual implementation.
    """

    def __init__(self, cfg=None, model_name=None, weight_path=None):
        if cfg:
            model_name = cfg['model_name']
            weight_path = cfg['weight_path']

            if model_name == 'yolov5':
                self.S1model = YOLOV5S(weights=weight_path)
                self.S1model.inference = self.yolov5_detect

            # ToDo
            elif model_name == 'faster_rcnn':
                self.S1model = YOLOV5S(weights=weight_path)
                self.S1model.inference = self.yolov5_detect
        else:
            if model_name == 'yolov5':
                self.S1model = YOLOV5S(weights=weight_path)
                self.S1model.inference = self.yolov5_detect

            # ToDo
            elif model_name == 'faster_rcnn':
                self.S1model = YOLOV5S(weights=weight_path)
                self.S1model.inference = self.yolov5_detect

    def yolov5_detect(self,
                      source='../example/source/',
                      save_dir='../res',
                      imgsz=(640, 640),
                      conf_thres=0.6,
                      iou_thres=0.45,
                      max_det=1000,
                      line_thickness=3,
                      hide_labels=True,
                      hide_conf=False,
                      ):

        color = Colors()
        detmodel = self.S1model
        stride, names = detmodel.stride, detmodel.names
        torch.no_grad()
        # Run inference
        if isinstance(source, np.ndarray):
            detmodel.eval()
            im = letterbox(source, imgsz, stride=stride, auto=True)[0]  # padded resize
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im).to(detmodel.device)
            im = im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

            # Inference
            pred = detmodel(im)
            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=False, max_det=max_det)
            # Process predictions
            for i, det in enumerate(pred):  # per image
                annotator = Annotator(source, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], source.shape).round()

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class

                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')

                        annotator.box_label(xyxy, label, color=color(c + 2, True))

                # Stream results
                im0 = annotator.result()
                # Save results (image with detections)
            return im0

        else:
            # Ensure the save directory exists
            os.makedirs(save_dir, exist_ok=True)
            dataset = LoadImages(source, img_size=imgsz, stride=stride)
            seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
            for path, im, im0s, s in dataset:
                im = torch.from_numpy(im).to(detmodel.device)
                im = im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

                # Inference
                pred = detmodel(im)
                # NMS
                pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=False, max_det=max_det)
                # Process predictions
                for i, det in enumerate(pred):  # per image
                    seen += 1
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                    p = Path(p)  # to Path
                    save_path = str(save_dir + p.name)  # im.jpg
                    s += '%gx%g ' % im.shape[2:]  # print string
                    annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, 5].unique():
                            n = (det[:, 5] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        for *xyxy, conf, cls in reversed(det):
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')

                            annotator.box_label(xyxy, label, color=color(c + 2, True))

                    # Stream results
                    im0 = annotator.result()
                    # Save results (image with detections)
                    if save_dir == 'buffer':
                        return im0
                    else:
                        cv2.imwrite(save_path, im0)
                        del im0  # Release memory after saving

            # Print results
            print(f"Results saved to {colorstr('bold', save_dir)}")

    #ToDo
    def faster_rcnn_detect(self,
                           source='../example/source/',
                           save_dir='../res',
                           weight_path='../example/detect/',
                           imgsz=(640, 640),
                           conf_thres=0.25,
                           iou_thres=0.45,
                           max_det=1000,
                           line_thickness=3,
                           hide_labels=False,
                           hide_conf=False,
    ):
        pass


def is_valid_file(path, total_ext):
    """
    Checks if the file has a valid extension.

    Parameters:
    - path (str): Path to the file.
    - total_ext (list): List of valid extensions.

    Returns:
    - bool: True if the file has a valid extension, False otherwise.
    """
    last_element = os.path.basename(path)
    if any(last_element.lower().endswith(ext) for ext in total_ext):
        return True
    else:
        return False


def path_to_tmp(original_path: str, prefix="/tmp/images_") -> str:
    """
    Convert any path into a unique /tmp path.
    Uses a SHA1 hash so the result is stable across runs.
    """
    abs_path = os.path.abspath(original_path)
    h = hashlib.sha1(abs_path.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}{h}"


def get_key_from_value(d, value):
    """
    Gets the key from a dictionary based on the value.

    Parameters:
    - d (dict): Dictionary.
    - value: Value to find the key for.

    Returns:
    - key: Key corresponding to the value, or None if not found.
    """
    if isinstance(d, list):
        return d[value]
    for key, val in d.items():
        if val == value:
            return key
    return None


def preprocess_image_yolo(im0, imgsz, stride, detmodel):
    im = letterbox(im0, imgsz, stride=stride, auto=True)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    im = torch.from_numpy(im).to(detmodel.device)
    im = im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    return im


def process_predictions_yolo(det, im, im0, names, line_thickness, hide_labels, hide_conf, color):
    annotator = Annotator(im0, line_width=line_thickness, example=str(names))
    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

        # Print results
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class

        for *xyxy, conf, cls in reversed(det):
            c = int(cls)  # integer class
            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')

            annotator.box_label(xyxy, label, color=color(c + 2, True))

    # Stream results
    im0 = annotator.result()
    return im0


# Usage-----------------------------------------------------------------------------------------------------------------
def main():

    """
    cfg = ''
    weight_path = ''

    source = ''
    save_path = ''
    test = Classify_Model(cfg=cfg, weight_path=weight_path)

    test.inference(source=source, save_path=save_path)
    # test.benchmark()
    """

    """
    source = ''
    weight_path = ''
    save_dir = ''
    test = Detection_Model(model_name='yolov5', weight_path=weight_path)
    test.yolov5_detect(source=source, save_dir=save_dir,)
    """


if __name__ == '__main__':
    main()
