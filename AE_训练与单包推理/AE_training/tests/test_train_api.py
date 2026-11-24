import inspect
import json
import os
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ae_train as m                      # noqa: E402
from pre_blocks import BLOCK_SAMPLES      # noqa: E402


def write_iq(path, n_samples, seed=0, scale=1000, byte_order='b'):
    """写一个 .iq：int16，I/Q 交错（byte_order 'b' 大端 / 'l' 小端）"""
    rng = np.random.default_rng(seed)
    dtype = '>i2' if byte_order == 'b' else '<i2'
    rng.integers(-scale, scale, size=n_samples * 2, dtype=np.int16).astype(dtype).tofile(path)


@pytest.fixture
def tiny_train(tmp_path, monkeypatch):
    """让 train() 只训练 1 轮、强制 CPU、产物放进临时目录；train() 本身没有这些参数"""
    real = m.default_options

    def fast():
        o = real()
        o.update(max_epoch=1, use_cpu=True)
        return o

    monkeypatch.setattr(m, 'default_options', fast)
    monkeypatch.setattr(m, 'MODELS_ROOT', str(tmp_path / 'models'))
    return tmp_path


def make_raw(tmp_path, n_blocks=20):
    raw = tmp_path / 'raw'
    raw.mkdir()
    write_iq(str(raw / 'jizhun.iq'), BLOCK_SAMPLES * n_blocks, seed=1)
    return raw


def test_train_takes_model_id_data_root_and_byte_order():
    assert list(inspect.signature(m.train).parameters) == ['model_id', 'data_root', 'byte_order', 'quantile']


def test_models_root_is_next_to_ae_train_py():
    assert os.path.isabs(m.MODELS_ROOT)
    assert m.MODELS_ROOT == os.path.join(os.path.dirname(os.path.abspath(m.__file__)), 'models')


def test_cli_defaults_untouched():
    """命令行参数和默认值一个都不能变"""
    opts = m.default_options()
    assert opts['out_dir'] == './output/run1'
    assert opts['data_dir'] == '/data/hdd1/project/anomaly_detection/dataset/ae_blocks'
    assert opts['block_quantile'] == 0.99            # 命令行是小数，start_test 才是百分数
    assert opts['max_epoch'] == 400
    assert 'eval' not in opts                      # 训练项目已不再有 --eval
    assert 'data_root' not in opts and 'model_id' not in opts


def test_train_empty_folder_leaves_no_blocks(tiny_train):
    """文件夹里没有任何 .iq：报错，模型文件夹里不留 blocks"""
    raw = tiny_train / 'raw'
    raw.mkdir()
    with pytest.raises(ValueError, match='没有任何 .iq'):
        m.train('003', str(raw))
    assert os.listdir(tiny_train / 'models') == [], '失败后临时目录必须被删掉，不留残留'


def test_train_bad_byte_order(tiny_train):
    raw = make_raw(tiny_train)
    with pytest.raises(ValueError, match='byte_order'):
        m.train('003', str(raw), 'x')


def test_train_all_files_invalid_leaves_no_blocks(tiny_train):
    """所有 .iq 都无效（空文件）：报错，留文件夹便于排查"""
    raw = tiny_train / 'raw'
    raw.mkdir()
    (raw / 'a.iq').write_bytes(b'')
    with pytest.raises(ValueError, match='全部无效'):
        m.train('003', str(raw))


def write_iq_bytes(path, data):
    """把已经生成好的 int16 数组按大端写进 .iq（用来把一段信号切成多个小包）"""
    np.asarray(data, dtype=np.int16).astype('>i2').tofile(path)


def test_train_small_packets_match_single_file(tmp_path):
    """同一段信号拆成多个小包与单个大文件，切块结果逐点一致"""
    from pre_blocks import load_blocks_from_dir

    n_total = BLOCK_SAMPLES * 20
    rng = np.random.default_rng(5)
    full = rng.integers(-1000, 1000, size=n_total * 2, dtype=np.int16)

    one = tmp_path / 'one'
    one.mkdir()
    write_iq_bytes(str(one / 'big.iq'), full)

    many = tmp_path / 'many'
    many.mkdir()
    per = BLOCK_SAMPLES * 5
    for i in range(4):                       # 每个小包 5 块，首尾相接
        write_iq_bytes(str(many / f'p{i}.iq'), full[i * per * 2:(i + 1) * per * 2])

    a = load_blocks_from_dir(str(one))
    b = load_blocks_from_dir(str(many))
    assert a.shape == (20, BLOCK_SAMPLES, 2)
    assert b.shape == (20, BLOCK_SAMPLES, 2)
    assert np.array_equal(a, b)


def test_load_blocks_from_dir_skips_bad_files(tmp_path):
    """空文件 / 奇数 int16 / 不足一块的文件被跳过，其余照常合并"""
    from pre_blocks import load_blocks_from_dir

    d = tmp_path / 'mix'
    d.mkdir()
    write_iq(str(d / 'ok1.iq'), BLOCK_SAMPLES * 3, seed=1)
    write_iq(str(d / 'ok2.iq'), BLOCK_SAMPLES * 2, seed=2)
    (d / 'empty.iq').write_bytes(b'')
    np.array([1, 2, 3], dtype='>i2').tofile(str(d / 'odd.iq'))
    write_iq(str(d / 'short.iq'), 1000, seed=3)

    blocks = load_blocks_from_dir(str(d))

    assert blocks.shape == (5, BLOCK_SAMPLES, 2)      # 只有两个好文件的 3+2 块


def test_load_blocks_from_dir_little_endian(tmp_path):
    """byte_order='l' 读小端文件，结果与同样内容的大端文件一致"""
    from pre_blocks import load_blocks_from_dir

    big = tmp_path / 'big'
    lit = tmp_path / 'lit'
    big.mkdir()
    lit.mkdir()
    write_iq(str(big / 'a.iq'), BLOCK_SAMPLES * 2, seed=7, byte_order='b')
    write_iq(str(lit / 'a.iq'), BLOCK_SAMPLES * 2, seed=7, byte_order='l')

    assert np.array_equal(load_blocks_from_dir(str(big), byte_order='b'),
                          load_blocks_from_dir(str(lit), byte_order='l'))


def test_train_refuses_existing_model_dir(tiny_train, monkeypatch):
    """同模型编号、同一秒重复调用：后到的必须报错，且不碰先到的产物"""
    raw = make_raw(tiny_train)
    monkeypatch.setattr(m.time, 'strftime', lambda fmt, *a: '20261014_153012')
    existing = tiny_train / 'models' / '003_20261014_153012'
    existing.mkdir(parents=True)
    (existing / 'marker.txt').write_text('first', encoding='utf-8')

    with pytest.raises(FileExistsError):
        m.train('003', str(raw))

    assert os.listdir(existing) == ['marker.txt'], '后到的调用不应往已有文件夹里写任何东西'
    assert (existing / 'marker.txt').read_text(encoding='utf-8') == 'first'


def test_train_end_to_end_tiny(tiny_train):
    """合成 20 块基准数据，训练 1 轮：流程走通、只留 3 样产物"""
    raw = make_raw(tiny_train)

    info = m.train('003', str(raw))

    d = info['model_dir']
    assert os.path.isabs(d)
    assert os.path.dirname(d) == str(tiny_train / 'models')
    assert info['model_id'] == '003'
    assert sorted(info) == ['center', 'model_dir', 'model_id', 'sigma', 'thr_sigma', 'threshold'], \
        '返回编号、路径、中心、σ、阈值和阈值换算'
    assert isinstance(info['threshold'], float)
    assert isinstance(info['center'], float)
    assert isinstance(info['sigma'], float)
    assert info['thr_sigma'] == pytest.approx(info['threshold'] / info['sigma'])
    assert re.fullmatch(r'003_\d{8}_\d{6}', os.path.basename(d)), '文件夹名仍在路径最后一级'

    # 训练成功后只剩测试要读的三样
    assert sorted(os.listdir(d)) == ['cal_block_err.npy', 'checkpoints', 'model_info.json']
    assert os.listdir(os.path.join(d, 'checkpoints')) == ['last.pth']
    # 20 块按 75/25 划分：15 训练 + 5 校准，不再有测试块
    assert np.load(os.path.join(d, 'cal_block_err.npy')).shape == (5,)


def test_train_failure_removes_tmp_and_keeps_existing_models(tiny_train, monkeypatch):
    """训练中途失败：临时目录（含预处理的 blocks/）被删除；已有模型目录原封不动"""
    raw = make_raw(tiny_train)
    existing = tiny_train / 'models' / '001_20200101_000000'
    existing.mkdir(parents=True)
    (existing / 'marker.txt').write_text('keep', encoding='utf-8')

    def boom(options):
        assert os.path.exists(os.path.join(options['data_dir'], 'meta.json')), '失败前 blocks 应已生成'
        raise RuntimeError('训练炸了')
    monkeypatch.setattr(m, 'main_worker', boom)

    with pytest.raises(RuntimeError, match='训练炸了'):
        m.train('003', str(raw))

    assert os.listdir(tiny_train / 'models') == ['001_20200101_000000']
    assert (existing / 'marker.txt').read_text(encoding='utf-8') == 'keep'


def test_train_keyboard_interrupt_also_cleans_up(tiny_train, monkeypatch):
    """Ctrl+C（KeyboardInterrupt 不是 Exception）也要清理临时目录"""
    raw = make_raw(tiny_train)

    def interrupted(options):
        raise KeyboardInterrupt
    monkeypatch.setattr(m, 'main_worker', interrupted)

    with pytest.raises(KeyboardInterrupt):
        m.train('003', str(raw))

    assert os.listdir(tiny_train / 'models') == []


def test_train_refuses_existing_tmp_dir_without_deleting_it(tiny_train, monkeypatch):
    """同编号同一秒另一次训练正在用临时目录：后到的报错，且不能把别人的临时目录删掉"""
    raw = make_raw(tiny_train)
    monkeypatch.setattr(m.time, 'strftime', lambda fmt, *a: '20261014_153012')
    busy = tiny_train / 'models' / '.003_20261014_153012.tmp'
    busy.mkdir(parents=True)
    (busy / 'marker.txt').write_text('running', encoding='utf-8')

    with pytest.raises(FileExistsError):
        m.train('003', str(raw))

    assert (busy / 'marker.txt').read_text(encoding='utf-8') == 'running'


def test_train_success_leaves_no_tmp_dir(tiny_train):
    raw = make_raw(tiny_train)
    info = m.train('003', str(raw))
    names = os.listdir(tiny_train / 'models')
    assert names == [os.path.basename(info['model_dir'])]
    assert not names[0].startswith('.')


# ---------- 百分位数输入 ----------

@pytest.mark.parametrize('bad', [0, 100, -1, 101, 'abc', '', None])
def test_train_bad_quantile(tiny_train, bad):
    raw = make_raw(tiny_train)
    with pytest.raises(ValueError):
        m.train('003', str(raw), quantile=bad)
    assert not (tiny_train / 'models').exists(), '参数非法时不应建模型文件夹'


def test_train_returns_threshold_of_given_quantile(tiny_train):
    """居中偏差：center = cal 中位数，threshold = (|cal - center| 的该百分位数 + 余量×σ) × 倍数"""
    raw = make_raw(tiny_train)
    info = m.train('003', str(raw), quantile=50)

    cal = np.load(os.path.join(info['model_dir'], 'cal_block_err.npy'))
    center = float(np.median(cal))
    dev = np.abs(cal - center)
    assert info['center'] == pytest.approx(center)
    assert info['sigma'] == pytest.approx(float(1.4826 * np.median(dev)))
    assert info['threshold'] == pytest.approx(2 * float(np.quantile(dev, 0.5))), '默认倍数 2、余量 0'

    with open(os.path.join(info['model_dir'], 'model_info.json'), encoding='utf-8') as f:
        meta = json.load(f)
    assert meta['block_quantile'] == pytest.approx(50), 'model_info 里记百分数'
    assert meta['threshold_margin'] == pytest.approx(0.0)
    assert meta['threshold_scale'] == pytest.approx(2.0)
    assert meta['cal_block_center'] == pytest.approx(info['center'])
    assert meta['cal_block_sigma'] == pytest.approx(info['sigma'])
    assert meta['cal_block_threshold'] == pytest.approx(info['threshold'])


# ---------- 阈值余量 / 倍数 ----------

def test_threshold_scale_doubles_threshold_and_thr_sigma(tiny_train):
    """默认倍数 2：输出阈值和 thr_sigma 都是原值的 2 倍"""
    raw = make_raw(tiny_train)
    info = m.train('003', str(raw), quantile=99)

    cal = np.load(os.path.join(info['model_dir'], 'cal_block_err.npy'))
    dev = np.abs(cal - info['center'])
    base = float(np.quantile(dev, 0.99))
    assert info['threshold'] == pytest.approx(2 * base)
    assert info['thr_sigma'] == pytest.approx(round(2 * base / info['sigma'], 4))


def test_threshold_margin_and_scale_can_be_disabled(tiny_train, monkeypatch):
    """余量 0 + 倍数 1：阈值完全回到旧口径"""
    raw = make_raw(tiny_train)
    real = m.default_options
    monkeypatch.setattr(m, 'default_options',
                        lambda: {**real(), 'threshold_margin': 0.0, 'threshold_scale': 1.0})
    info = m.train('003', str(raw), quantile=95)

    cal = np.load(os.path.join(info['model_dir'], 'cal_block_err.npy'))
    dev = np.abs(cal - info['center'])
    assert info['threshold'] == pytest.approx(float(np.quantile(dev, 0.95)))


@pytest.mark.parametrize('bad', [-0.1, float('nan'), float('inf')])
def test_threshold_margin_rejects_negative_and_non_finite(tiny_train, monkeypatch, bad):
    """负余量和 NaN/inf 必须在建任何东西之前就报错"""
    raw = make_raw(tiny_train)
    real = m.default_options
    monkeypatch.setattr(m, 'default_options',
                        lambda: {**real(), 'threshold_margin': bad})
    with pytest.raises(ValueError, match='余量'):
        m.train('003', str(raw))
    assert not (tiny_train / 'models').exists()


@pytest.mark.parametrize('bad', [0.0, -1.0, float('nan'), float('inf')])
def test_threshold_scale_rejects_non_positive_or_non_finite(tiny_train, monkeypatch, bad):
    """倍数必须是 > 0 的有限数，同样在建目录之前报错"""
    raw = make_raw(tiny_train)
    real = m.default_options
    monkeypatch.setattr(m, 'default_options',
                        lambda: {**real(), 'threshold_scale': bad})
    with pytest.raises(ValueError, match='倍数'):
        m.train('003', str(raw))
    assert not (tiny_train / 'models').exists()


def test_compute_threshold_applies_stored_margin(tmp_path):
    """compute_threshold 读模型里的余量；旧模型没有这个键时按 0 处理"""
    d = tmp_path / 'model'
    d.mkdir()
    cal = np.array([1.0, 2.0, 3.0, 4.0])
    np.save(d / 'cal_block_err.npy', cal)
    center = float(np.median(cal))
    sigma = float(1.4826 * np.median(np.abs(cal - center)))

    thr0, _, _ = m.compute_threshold(75, str(d), 'x')
    assert thr0 == pytest.approx(float(np.quantile(np.abs(cal - center), 0.75)))

    (d / 'model_info.json').write_text(json.dumps({'threshold_margin': 0.5}), encoding='utf-8')
    thr1, _, thr_sigma = m.compute_threshold(75, str(d), 'x')
    assert thr1 == pytest.approx(thr0 + 0.5 * sigma)
    assert thr_sigma == pytest.approx(round(thr1 / sigma, 4))


def test_compute_threshold_applies_stored_scale(tmp_path):
    """compute_threshold 也乘模型里的倍数；旧模型没有这个键时按 1.0 处理"""
    d = tmp_path / 'model'
    d.mkdir()
    cal = np.array([1.0, 2.0, 3.0, 4.0])
    np.save(d / 'cal_block_err.npy', cal)
    center = float(np.median(cal))
    sigma = float(1.4826 * np.median(np.abs(cal - center)))
    base = float(np.quantile(np.abs(cal - center), 0.8))

    thr0, _, _ = m.compute_threshold(80, str(d), 'x')
    assert thr0 == pytest.approx(base), '缺倍数键 → 1.0'

    (d / 'model_info.json').write_text(json.dumps({'threshold_scale': 2.0}), encoding='utf-8')
    thr1, _, thr_sigma = m.compute_threshold(80, str(d), 'x')
    assert thr1 == pytest.approx(2 * base)
    assert thr_sigma == pytest.approx(round(thr1 / sigma, 4))


# ---------- compute_threshold ----------

def test_compute_threshold_reads_cal_err_only(tmp_path):
    """只读 cal_block_err.npy 算居中偏差阈值，返回 (阈值, 编号)，不改任何文件"""
    d = tmp_path / 'model'
    d.mkdir()
    cal = np.array([1.0, 2.0, 3.0, 4.0])
    np.save(d / 'cal_block_err.npy', cal)
    before = sorted(os.listdir(d)), (d / 'cal_block_err.npy').stat().st_mtime_ns

    thr, mid, thr_sigma = m.compute_threshold(50, str(d), '007')

    center = float(np.median(cal))                      # = 2.5
    assert thr == pytest.approx(float(np.quantile(np.abs(cal - center), 0.5)))
    sigma = float(1.4826 * np.median(np.abs(cal - center)))
    assert thr_sigma == pytest.approx(thr / sigma), '第三项是阈值折合的 σ 倍数'
    assert mid == '007'
    assert before == (sorted(os.listdir(d)), (d / 'cal_block_err.npy').stat().st_mtime_ns), '不应写入或改动文件'


@pytest.mark.parametrize('q, expected', [(50, 0.5), (99, 0.99), ('95', 0.95), ('90%', 0.90)])
def test_compute_threshold_accepts_percent_forms(tmp_path, q, expected):
    d = tmp_path / 'model'
    d.mkdir()
    cal = np.arange(1, 101, dtype=float)
    np.save(d / 'cal_block_err.npy', cal)

    thr, _, thr_sigma = m.compute_threshold(q, str(d), 'x')

    center = float(np.median(cal))                      # = 50.5
    assert thr == pytest.approx(float(np.quantile(np.abs(cal - center), expected)))
    sigma = float(1.4826 * np.median(np.abs(cal - center)))
    assert thr_sigma == pytest.approx(thr / sigma)


def test_compute_threshold_prefers_center_from_model_info(tmp_path):
    """model_info.json 里有 cal_block_center 时优先用它（与现算的中位数可能不同）"""
    d = tmp_path / 'model'
    d.mkdir()
    cal = np.array([1.0, 2.0, 3.0, 4.0])
    np.save(d / 'cal_block_err.npy', cal)
    (d / 'model_info.json').write_text(json.dumps({'cal_block_center': 2.0}), encoding='utf-8')

    thr, _, thr_sigma = m.compute_threshold(50, str(d), 'x')

    assert thr == pytest.approx(float(np.quantile(np.abs(cal - 2.0), 0.5)))


def test_compute_threshold_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match='cal_block_err'):
        m.compute_threshold(99, str(tmp_path / 'nope'), 'x')


@pytest.mark.parametrize('bad', [0, 100, 'abc', None])
def test_compute_threshold_bad_quantile(tmp_path, bad):
    d = tmp_path / 'model'
    d.mkdir()
    np.save(d / 'cal_block_err.npy', np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        m.compute_threshold(bad, str(d), 'x')
