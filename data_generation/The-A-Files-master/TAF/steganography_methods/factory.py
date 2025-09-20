from typing import Dict, List

from TAF.models.SteganographyMethod import SteganographyMethod
from TAF.models.types import MethodType
from TAF.steganography_methods.BlindSvdMethod import BlindSvdMethod
from TAF.steganography_methods.DctB1Method import DctB1Method
from TAF.steganography_methods.DctDeltaLsbMethod import DctDeltaLsbMethod
from TAF.steganography_methods.DsssMethod import DsssMethod
from TAF.steganography_methods.DwtLsbMethod import DwtLsbMethod
from TAF.steganography_methods.EchoMethod import EchoMethod
from TAF.steganography_methods.FsvcMethod import FsvcMethod
from TAF.steganography_methods.ImprovedPhaseCodingMethod import ImprovedPhaseCodingMethod
from TAF.steganography_methods.LsbMethod import LsbMethod
from TAF.steganography_methods.LwtMethod import LwtMethod
from TAF.steganography_methods.NormSpaceMethod import NormSpaceMethod
from TAF.steganography_methods.PatchworkMultilayerMethod import PatchworkMultilayerMethod
from TAF.steganography_methods.PhaseCodingMethod import PhaseCodingMethod
from TAF.steganography_methods.PrimeFactorInterpolatedMethod import PrimeFactorInterpolatedMethod
from TAF.steganography_methods.wavmark import WavMarkMethod
from TAF.steganography_methods.audioseal import AudioSealMethod
from TAF.steganography_methods.timbre import TimbreMethod
# from TAF.steganography_methods.silentcipher import SilentCipherMethod
# from TAF.steganography_methods.robustDNN import RobustDNNMethod
class SteganographyMethodFactory:

    @staticmethod
    def get(sr: int, methodType: MethodType) -> SteganographyMethod:
        return SteganographyMethodFactory._all_methods(sr).get(methodType)

    @staticmethod
    def get_all(sr: int) -> List[SteganographyMethod]:
        return [value for _, value in SteganographyMethodFactory._all_methods(sr).items()]

    @staticmethod
    def _all_methods(sr: int) -> Dict[MethodType, SteganographyMethod]:
        return {

            #phase1 train & eval
            #Handcrafted
            MethodType.BLIND_SVD_METHOD: BlindSvdMethod(), #svd
            MethodType.IMPROVED_PHASE_CODING_METHOD: ImprovedPhaseCodingMethod(), #phase
            MethodType.DWT_LSB_METHOD: DwtLsbMethod(), #lsb
            MethodType.FSVC_METHOD: FsvcMethod(sr=sr), #fsvc
            MethodType.PATCHWORK_MULTILAYER_METHOD: PatchworkMultilayerMethod(sr=sr), #patch
            MethodType.LWT_METHOD: LwtMethod(), #lwt
            # #DNN
            # MethodType.WAVMARK_METHOD: WavMarkMethod(), #wavmark
            # MethodType.AUDIOSEAL_METHOD: AudioSealMethod(), #audioseal
            # MethodType.TIMBRE_METHOD: TimbreMethod(), #timbre

            #phase2 eval

            #Handcrafted
            MethodType.NORM_SPACE_METHOD: NormSpaceMethod(sr=sr), #norm
            MethodType.DSSS_METHOD: DsssMethod(), #dsss
            MethodType.PRIME_FACTOR_INTERPOLATE: PrimeFactorInterpolatedMethod(),
            MethodType.ECHO_METHOD: EchoMethod(), #echo

            #DNN
            # MethodType.CIPHER_METHOD: SilentCipherMethod(), #silent
            # MethodType.RobustDNN_METHOD: RobustDNNMethod(), #dnn


            #For Future use
            # MethodType.DCT_B1_METHOD: DctB1Method(sr=sr),
            # MethodType.DCT_DELTA_LSB_METHOD: DctDeltaLsbMethod(sr=sr),
            # MethodType.LSB_METHOD: LsbMethod(),

            #testing single watermarks
            # MethodType.PHASE_CODING_METHOD: PhaseCodingMethod(),
            # MethodType.PRIME_FACTOR_INTERPOLATE: PrimeFactorInterpolatedMethod(),
            # MethodType.DCT_B1_METHOD: DctB1Method(sr=sr), #dctb1
            
            
            
            
        }


