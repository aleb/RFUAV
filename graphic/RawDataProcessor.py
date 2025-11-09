# Tool to process raw data
import matplotlib.pyplot as plt
from scipy.signal import stft, windows
import numpy as np
import os
from io import BytesIO
import imageio
from PIL import Image
from typing import Union
from scipy.fft import fft
from scipy.io import wavfile

SAMPLES_FREQUENCY = 20e6


class RawDataProcessor:
    """transform raw data into images, video, and save the result locally
    func:
    - TransRawDataintoSpectrogram() can process raw data in batches and save the results locally
    - TransRawDataintoVideo() can transform raw data into video and save the result locally
    - ShowSpectrogram() can show the spectrum of the raw data in prev 0.1s to check the raw data quicly
    """

    def TransRawDataintoSpectrogram(self,
                                    fig_save_path: str,
                                    data_path: str,
                                    sample_rate: Union[int, float] = 100e6,
                                    stft_point: int = 2048,
                                    duration_time: float = 0.1,
                                    file_type=np.float32
                                    ):
        """transform the raw data into spectromgrams and save the results locally
        :param fig_save_path: the target dir path to save the image result.
        :param data_path: the input raw data dir path.
        :param sample_rate: the simple rate when collect the raw frequency data using USRP, ref: https://en.wikipedia.org/wiki/Sampling_(signal_processing)
        :param stft_point: the STFT points using in STFT transformation, you better set this to 2**n (n is a Non-negative integer).
        :param duration_time: the duration time of single spectromgram.
        """
        DrawandSave(fig_save_path=fig_save_path, file_path=data_path, fs=sample_rate,
                    stft_point=stft_point, duration_time=duration_time, file_type=file_type)

    def TransRawDataintoVideo(self,
                              save_path: str,
                              data_path: str,
                              sample_rate: Union[int, float] = 100e6,
                              stft_point: int = 2048,
                              duration_time: float = 0.1,
                              fps: int = 5,
                              file_type=np.float32
                              ):
        """transform the raw data into video and save the result locally
        :param save_path: the target dir path to save the image result.
        :param data_path: the input raw data dir path.
        :param sample_rate: the simple rate when collect the raw frequency data using USRP, ref: https://en.wikipedia.org/wiki/Sampling_(signal_processing)
        :param stft_point: the STFT points using in STFT transformation, you better set this to 2**n (n is a Non-negative integer). ref: https://en.wikipedia.org/wiki/Short-time_Fourier_transform#Inverse_STFT
        :param duration_time: the duration time of single spectromgram.
        :param fps: control the fps of generated video.
        """
        save_as_video(datapack=data_path, save_path=save_path, fs=sample_rate,
                      stft_point=stft_point, duration_time=duration_time, fps=fps, file_type=file_type)

    def ShowSpectrogram(self,
                        data_path: str,
                        drone_name: str = 'test',
                        sample_rate: Union[int, float] = 100e6,
                        stft_point: int = 2048,
                        duration_time: float = 0.1,
                        oneside: bool = False,
                        Middle_Frequency: float = 2400e6,
                        file_type=np.float32
                        ):
        """tool used to observe the spectrograms from a local datapack
        :param save_path: the target dir path to save the image result.
        :param data_path: the input raw data file path.
        :param sample_rate: the simple rate when collect the raw frequency data using USRP, ref: https://en.wikipedia.org/wiki/Sampling_(signal_processing)
        :param stft_point: the STFT points using in STFT transformation, you better set this to 2**n (n is a Non-negative integer). ref: https://en.wikipedia.org/wiki/Short-time_Fourier_transform#Inverse_STFT
        :param duration_time: the duration time of single spectromgram.
        :param oneside: set 'True' if you want to observe the real & imaginary parts separately, set 'False' to show the complete spectrogram
        :param Middle_Frequency: the middle frequency set to collect data using USRP in the frequency band, ref: https://en.wikipedia.org/wiki/Center_frequency
        """

        if oneside:
            show_half_only(datapack=data_path, drone_name=drone_name,
                           fs=sample_rate, stft_point=stft_point, duration_time=duration_time, file_type=file_type)

        else:
            show_spectrum(datapack=data_path, drone_name=drone_name, fs=sample_rate, stft_point=stft_point,
                           duration_time=duration_time, Middle_Frequency=Middle_Frequency, file_type=file_type)


def generate_images(datapack: str = None,
                    file: str = None,
                    pack: str = None,
                    fs: int = SAMPLES_FREQUENCY,
                    stft_point: int = 1024,
                    duration_time: float = 0.1,
                    ratio: int = 1,  # 控制产生图片时间间隔的倍率，默认为1生成视频的倍率
                    location: str = 'buffer',
                    ):
    """
    Generates images from the given data using Short-Time Fourier Transform (STFT).

    Parameters:
    - datapack (str): Path to the data file.
    - file (str): File name.
    - pack (str): Pack name.
    - fs (int): Sampling frequency, default is 100 MHz. ref: https://en.wikipedia.org/wiki/Sampling_(signal_processing)
    - stft_point (int): Number of points for STFT, default is 1024. ref: https://en.wikipedia.org/wiki/Short-time_Fourier_transform#Inverse_STFT
    - duration_time (float): Duration time for each image, default is 0.1 seconds.
    - ratio (int): Controls the time interval ratio for generating images 2^(-ratio). For example 1->0.5s.
    - location (str): Location to save the images, default is 'buffer'.

    Returns:
    - list: List of images if `location` is 'buffer'.
    """
    slice_point = int(fs * duration_time)
    if datapack.endswith('.wav'):
        # The returned 'data' is composed of tuples of float32 values.
        _sampling_rate, data = wavfile.read(datapack)
        fs = _sampling_rate
        # This also works:
        # np.complex64 means both the real and imaginary parts are stored as 32-bit (single-precision) floating-point numbers.
        # data1 = data.flatten().view(np.complex64)
        I = data[:, 0]
        Q = data[:, 1]
        data = I + 1j * Q
    else:
        # Load an array of float32
        data = np.fromfile(datapack, dtype=np.int16)
        # Pair the values in two, as complex numbers
        data = data[::2] + data[1::2] * 1j

    if location == 'buffer':
        images = []
    else:
        if os.path.exists(location):
            return
        os.mkdir(location)

    i = 0.
    while (i + 1) * slice_point <= len(data):
        start = int(i * slice_point)
        end = int((i + 1) * slice_point)
        f, t, Zxx = STFT(data[start:end],
                         stft_point=stft_point, fs=fs, duration_time=duration_time, onside=False)
        f = np.fft.fftshift(f)
        Zxx = np.fft.fftshift(Zxx, axes=0)
        aug = 10 * np.log10(np.abs(Zxx))
        extent = [t.min(), t.max(), f.min(), f.max()]

        plt.figure()
        plt.imshow(aug, extent=extent, aspect='auto', origin='lower', cmap='jet')
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=None, hspace=None)

        if location == 'buffer':
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300)
            plt.close()

            buffer.seek(0)
            images.append(Image.open(buffer))

        else:
            # plt.savefig(location + (file + '/' if file else '') + (pack + '/' if pack else '') + file + ' (' + str(i) + ').jpg', dpi=300)
            plt.savefig(os.path.join(location, '{:08}.jpg'.format(i)), dpi=300)
            plt.close()

        i += 2 ** (-ratio)

    if location == 'buffer':
        return images


def save_as_video(datapack: str,
                  save_path: str,
                  fs: int = 100e6,
                  stft_point: int = 1024,
                  duration_time: float = 0.1,
                  fps: int = 5,  # 视频帧率
                  file_type=np.float32
                  ):
    """
    Saves the generated images as a video.

    Parameters:
    - datapack (str): Path to the data file.
    - save_path (str): Path to save the video.
    - fs (int): Sampling frequency, default is 100 MHz.
    - stft_point (int): Number of points for STFT, default is 1024.
    - duration_time (float): Duration time for each segment, default is 0.1 seconds.
    - fps (int): Frame rate of the video, default is 5.
    """

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(datapack):
        raise ValueError('File not found!')

    images = generate_images(datapack=datapack, fs=fs, stft_point=stft_point, duration_time=duration_time, file_type=file_type)
    imageio.mimsave(save_path+'video.mp4', images, fps=fps)


def show_spectrum(datapack: str = '',
                  drone_name: str = 'test',
                  fs: int = 100e6,
                  stft_point: int = 2048,
                  duration_time: float = 0.1,
                  Middle_Frequency: float = 2400e6,
                  file_type=np.float32
                  ):

    """
    Displays the spectrum of the given data.

    Parameters:
    - datapack (str): Path to the data file.
    - drone_name (str): Name of the drone, default is 'test'.
    - fs (int): Sampling frequency, default is 100 MHz.
    - stft_point (int): Number of points for STFT, default is 2048.
    - duration_time (float): Duration time for each segment, default is 0.1 seconds.
    - Middle_Frequency (float): Middle frequency, default is 2400 MHz.
    """

    with open(datapack, 'rb') as fp:
        print("reading raw data...")
        read_data = np.fromfile(fp, dtype=file_type)

        data = read_data[::2] + read_data[1::2] * 1j
        print('STFT transforming')

        f, t, Zxx = STFT(data, stft_point=stft_point, fs=fs, duration_time=duration_time, onside=False)
        f = np.linspace(Middle_Frequency-fs / 2, Middle_Frequency+fs / 2, stft_point)
        Zxx = np.fft.fftshift(Zxx, axes=0)

        plt.figure()
        aug = 10 * np.log10(np.abs(Zxx))
        extent = [t.min(), t.max(), f.min(), f.max()]
        plt.imshow(aug, extent=extent, aspect='auto', origin='lower')
        plt.colorbar()
        plt.title(drone_name)
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.show()


def show_half_only(datapack: str = '',
                   drone_name: str = 'test',
                   fs: int = 100e6,
                   stft_point: int = 2048,
                   duration_time: float = 0.1,
                   file_type=np.float32
                   ):

    """
    Displays I and Q components of the given data separately.

    Parameters:
    - datapack (str): Path to the data file.
    - drone_name (str): Name of the drone, default is 'test'.
    - fs (int): Sampling frequency, default is 100 MHz.
    - stft_point (int): Number of points for STFT, default is 2048.
    - duration_time (float): Duration time for each segment, default is 0.1 seconds.
    """

    with open(datapack, 'rb') as fp:
        print("reading raw data...")
        read_data = np.fromfile(fp, dtype=file_type)
        dataI = read_data[::2]
        dataQ = read_data[1::2]

        f_I, t_I, Zxx_I = STFT(dataI, fs=fs, stft_point=stft_point, duration_time=duration_time)
        f_Q, t_Q, Zxx_Q = STFT(dataQ, fs=fs, stft_point=stft_point, duration_time=duration_time)

        # I部分數據的時頻圖
        print('Drawing')
        plt.figure()
        aug_I = 10 * np.log10(np.abs(Zxx_I))
        plt.pcolormesh(t_I, f_I, np.abs(aug_I))
        plt.title(drone_name + " I")
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.colorbar()
        plt.show()
        print("figure I done")

        # Q部分數據的時頻圖
        plt.figure()
        aug_Q = 10 * np.log10(np.abs(Zxx_Q))
        plt.pcolormesh(t_Q, f_Q, np.abs(aug_Q), cmap='jet')
        plt.title(drone_name + " Q")
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.colorbar()
        plt.show()
        print("figure Q done")


def DrawandSave(
        fig_save_path: str,
        file_path: str,
        fs: int = 100e6,
        stft_point: int = 2048,
        duration_time: float = 0.1,
        file_type=np.float32
):

    """
    Draw and save the images from the given data files.

    Parameters:
    - fig_save_path (str): Path to save the figures.
    - file_path (str): Path to the data files.
    - fs (int): Sampling frequency, default is 100 MHz.
    - stft_point (int): Number of points for STFT, default is 2048.
    - duration_time (float): Duration time for each segment, default is 0.1 seconds.

    Your raw data should organize like this:
    file_path
        Drone 1
            data pack1.iq
            data pack2.iq
            ...
            data packn.iq
        Drone 2
            data pack1.iq
            data pack2.iq
            ...
            data packn.iq
        Drone 3
            data pack1.iq
            data pack2.iq
            ...
            data packn.iq
        .....
        Drone n
            ...
    """
    re_files = os.listdir(file_path)

    for file in re_files:
        packlist = os.listdir(os.path.join(file_path, file))
        for pack in packlist:
            check_folder(os.path.join(fig_save_path, file, pack))
            generate_images(datapack=os.path.join(file_path, file, pack),
                            file=file,
                            pack=pack,
                            fs=fs,
                            stft_point=stft_point,
                            duration_time=duration_time,
                            ratio=0,
                            location=fig_save_path,
                            file_type=file_type
                            )

            print(pack + ' Done')
        print(file + ' Done')
    print('All Done')


def check_folder(folder_path):
    """
    Checks and creates the folder if it does not exist.

    Parameters:
    - folder_path (str): Path to the folder.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"folder '{folder_path}' created。")
    else:
        print(f"folder '{folder_path}' existed。")


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


def waterfall_spectrogram(datapack, fft_size, fs, location, time_scale):
    """
    Generate and save waterfall spectrograms.

    This function reads a data pack, performs Fourier Transform to generate spectrograms,
    and saves the results as images based on the specified parameters.
    If location is set to 'buffer', images are saved in memory; otherwise, they are saved to the specified folder.

    Parameters:
    - datapack: Path to the data pack.
    - fft_size: Window size for Fast Fourier Transform.
    - fs: Sampling rate.
    - location: Image save location, can be 'buffer' (in memory) or a file system path.
    - time_scale: Time scale to control when to start scrolling the spectrogram.

    Returns:
    - images: A list of saved images when location is 'buffer'; otherwise, returns None.
    """
    if isinstance(datapack, str):
        data = np.fromfile(datapack, dtype=np.float32)
        data = data[::2] + data[1::2] * 1j
    if isinstance(datapack, np.ndarray):
        data = datapack
    pack_gap = 0
    j = 0
    gap = 150
    window = np.hanning(fft_size)
    spectrogram = []
    num_frames = len(data) // fft_size

    if location == 'buffer':
        images = []
    else:
        if not os.path.exists(location):
            os.makedirs(location)

    for i in range(num_frames):
        frame_data = data[i * fft_size: (i+1)*fft_size] * window
        magnitude = np.abs(np.fft.fftshift(fft(frame_data)))

        if i > time_scale:
            spectrogram = spectrogram[1:]
            spectrogram = np.concatenate((spectrogram, magnitude.reshape(1, fft_size)), axis=0)

        else:
            spectrogram.append(magnitude)

        pack_gap += 1
        if i == time_scale:
            spectrogram = np.array(spectrogram)
            plt.figure()
            plt.imshow(np.log10(spectrogram.T), aspect='auto', cmap='jet', origin='lower',
                       extent=[0, num_frames * (fft_size) / fs, -fs / 2, fs / 2])
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=None, hspace=None)
            plt.axis('off')

            if location != 'buffer':
                # 保存瀑布图
                plt.savefig(os.path.join(location, str(j) + 'waterfall_spectrogram.jpg'), dpi=300)
                plt.close()
            else:
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=300)
                plt.close()
                buffer.seek(0)
                images.append(BytesIO(buffer.getvalue()))

            j += 1
            pack_gap = 0

        if i > time_scale and pack_gap == gap:
            plt.figure()
            plt.imshow(np.log10(spectrogram.T), aspect='auto', cmap='jet', origin='lower',
                       extent=[0, num_frames * (fft_size) / fs, -fs / 2, fs / 2])
            plt.axis('off')
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=None, hspace=None)

            if location != 'buffer':
                plt.savefig(os.path.join(location, str(j) + 'waterfall_spectrogram.jpg'), dpi=300)
                plt.close()
            else:
                plt.savefig(buffer, format='png', dpi=300)
                plt.close()
                buffer.seek(0)
                images.append(BytesIO(buffer.getvalue()))
            j += 1
            pack_gap = 0

    return images


# Usage-----------------------------------------------------------------------------------------------------------------
def main():

    """
    data_path = data_path
    save_path = save_path
    test = RawDataProcessor()
    test.TransRawDataintoSpectrogram(fig_save_path=save_path,
                                     data_path=data_path,
                                     sample_rate=100e6,
                                     stft_point=1024,
                                     duration_time=0.1,
                                     )
    """

    """
    data_path = ''
    test.ShowSpectrogram(data_path=data_path,
                         drone_name='DJ FPV COMBO',
                         sample_rate=100e6,
                         stft_point=2048,
                         duration_time=0.1,
                         Middle_Frequency=2400e6
                         )
    """

    """
    save_path = ''
    data_path = ''
    test.TransRawDataintoSpectrogram(fig_save_path=save_path,
                                 data_path=data_path,
                                 sample_rate=100e6,
                                 stft_point=2048,
                                 duration_time=0.1,
                                 )
    """

    """
    datapack = ''
    save_path = ''
    save_as_video(datapack=datapack,
                  save_path=save_path,
                  fs=100e6,
                  stft_point=1024,
                  duration_time=0.1,
                  fps=5,
                  )

    show_spectrum(datapack=datapack,
                  drone_name='test',
                  fs=100e6,
                  stft_point=2048,
                  duration_time=0.1,
                  )
    show_half_only(datapack,
                   drone_name='test',
                   fs=100e6,
                   stft_point=2048,
                   duration_time=0.1,
                   )
    """


if __name__ == '__main__':
    main()
