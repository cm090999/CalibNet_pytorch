import numpy as np

def get_stats(error_data: np.ndarray, logger = None):

    ang_err_mean = np.degrees(np.mean(error_data[:,:3],axis=0))
    ang_err_std = np.degrees(np.std(error_data[:,:3],axis=0))
    ang_err_med = np.degrees(np.median(error_data[:,:3],axis=0))
    ang_err_95 = np.degrees(np.percentile(error_data[:,:3],axis=0, q=5))
    ang_err = np.degrees(np.mean(np.linalg.norm(error_data[:,:3],axis=1)))

    tsl_err_mean = np.mean(error_data[:,3:],axis=0)
    tsl_err_std = np.std(error_data[:,3:],axis=0)
    tsl_err_med = np.median(error_data[:,3:],axis=0)
    tsl_err_95 = np.percentile(error_data[:,3:],axis=0, q=5)
    tsl_err = np.mean(np.linalg.norm(error_data[:,3:],axis=1))

    logger.info('Angle error Total Mean (deg): {:.4f}'.format(ang_err))
    logger.info('Translation error Total Mean (m): {:.4f}'.format(tsl_err))

    logger.info('Angle error Mean (deg): X:{:.4f},Y:{:.4f},Z:{:.4f}'.format(*ang_err_mean))
    logger.info('Translation error Mean (m): X:{:.4f},Y:{:.4f},Z:{:.4f}'.format(*tsl_err_mean))

    logger.info('Angle error STD (deg): X:{:.4f},Y:{:.4f},Z:{:.4f}'.format(*ang_err_std))
    logger.info('Translation error STD (m): X:{:.4f},Y:{:.4f},Z:{:.4f}'.format(*tsl_err_std))

    logger.info('Angle error Median (deg): X:{:.4f},Y:{:.4f},Z:{:.4f}'.format(*ang_err_med))
    logger.info('Translation error Median (m): X:{:.4f},Y:{:.4f},Z:{:.4f}'.format(*tsl_err_med))

    logger.info('Angle error 95th Percentile (deg): X:{:.4f},Y:{:.4f},Z:{:.4f}'.format(*ang_err_95))
    logger.info('Translation error 95th Percentile (m): X:{:.4f},Y:{:.4f},Z:{:.4f}'.format(*tsl_err_95))