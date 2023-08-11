import argparse
import warnings
import os
import datetime
import time
import shutil
from collections import Iterable

'''
这里要尽量别import和实验环境有关的东西，比如pytorch之类的
'''


class Tmux_line():
    '''
    get error:  error connecting to /tmp/tmux-11502/default (No such file or directory)
    I meet this error when using this tmux tool in virtual environment (machine by rlaunch or zsh).
    The reason is that the directory /tmp/ is empty in the virtual environment.

    '''

    @classmethod
    def new_session(cls, session_name, first_window_name='first'):
        # '''  tmux new-session -s a -n editor -d
        # test:  new_session('a','b')
        # '''
        os.system("tmux new-session -s %s -n %s -d" % (session_name, first_window_name))

    @classmethod
    def new_window(cls, session_name, window_name):
        # '''  tmux neww -a -n tool -t init
        # test:  new_session('a','b')  & new_window('a', 'c')
        # '''
        os.system("tmux neww -a -n %s -t %s" % (window_name, session_name))

    @classmethod
    def switch_window(cls, session_name, window_name):
        # ''' tmux attach -t [session_name]  这个暂时还是别用，会从python弹到tmux对应窗口里面的
        # test:  new_session('a','b')  & new_window('a', 'c') & new_window('a', 'd') & switch_window('a', 'b')
        # '''
        os.system("tmux attach -t %s:%s" % (session_name, window_name))

    @classmethod
    def split_window(cls, session_name, window_name, h_v='h', panel_number=0):
        # ''' tmux split-window -h -t development
        # h表示横着分, v表示竖着分
        # test:  new_session('a','b')  & new_window('a', 'c') & split_window('a', 'b', h_v='h', panel_number=0)
        # '''
        assert h_v in ['h', 'v']
        os.system("tmux split-window -%s -t %s:%s.%s" % (h_v, session_name, window_name, panel_number))

    @classmethod
    def split_window_by_2(cls, session_name, window_name):
        # ''' 拆成4个panel '''
        cls.split_window(session_name, window_name, h_v='v', panel_number=0)  # 拆称0和1，横着

    @classmethod
    def split_window_by_4(cls, session_name, window_name):
        # ''' 拆成4个panel '''
        cls.split_window(session_name, window_name, h_v='h', panel_number=0)  # 拆称0和1，横着
        cls.split_window(session_name, window_name, h_v='v', panel_number=0)
        cls.split_window(session_name, window_name, h_v='v', panel_number=1)

    @classmethod
    def split_window_by_8(cls, session_name, window_name):
        # ''' 先拆成4个panel '''
        cls.split_window_by_4(session_name, window_name)
        for i in range(4):
            cls.split_window(session_name, window_name, h_v='v', panel_number=i)

    @classmethod
    def split_window_by_16(cls, session_name, window_name):
        # ''' 先拆成8个panel '''
        cls.split_window_by_8(session_name, window_name)
        for i in range(8):
            cls.split_window(session_name, window_name, h_v='h', panel_number=i)

    @classmethod
    def run_command(cls, session_name, window_name, panel_number=0, command_line='ls'):
        com = "tmux send-keys -t %s:%s.%s '%s' C-m" % (session_name, window_name, panel_number, command_line)
        # print(com)
        os.system(com)

    @classmethod
    def demo(cls):
        # tmux kill-session -t a
        # demo()
        session_name = 'k'
        window_name = 'c'
        cls.new_session(session_name)
        cls.new_window(session_name, window_name)
        cls.split_window_by_16(session_name, window_name)
        for i in range(16):
            time.sleep(0.1)
            cls.run_command(session_name, window_name, i, command_line='ls')

    @classmethod
    def demo_run_commands(cls):
        session_name = 's'
        line_ls = ['ls' for i in range(17)]
        cls.run_task(task_ls=line_ls, task_name='demo', session_name=session_name)

    @classmethod
    def run_command_v2(cls, session_name, window_name, panel_number=0, command_line='ls', **kwargs):
        for i in kwargs.keys():
            command_line += ' --%s %s' % (i, kwargs[i])
        cls.run_command(session_name, window_name, panel_number=panel_number, command_line=command_line)

    @classmethod
    def run_task(cls, task_ls, task_name='demo', session_name='k'):
        # task_ls is a list that contains some string line. Each string line is a command line we want to run.
        N = len(task_ls)
        window_number = 0
        ind = -1

        def create_window(window_number_, panel_number=16):
            window_name = task_name + '_%s' % window_number_
            cls.new_window(session_name, window_name)
            # cls.new_window(session_name, window_name)
            if panel_number == 16:
                print('create a window with %s panels' % panel_number)
                cls.split_window_by_16(session_name, window_name)
            elif panel_number == 8:
                print('create a window with %s panels' % panel_number)
                cls.split_window_by_8(session_name, window_name)
            elif panel_number == 4:
                print('create a window with %s panels' % panel_number)
                cls.split_window_by_4(session_name, window_name)
            elif panel_number == 2:
                print('create a window with %s panels' % panel_number)
                cls.split_window_by_2(session_name, window_name)
            elif panel_number == 1:
                print('create a window with %s panels' % panel_number)
            else:
                pass
            window_number_ += 1
            return window_number_, window_name

        def run_16(data_ls, cnt, window_number_):
            for i in range(len(data_ls) // 16):
                # create window
                window_number_, window_name = create_window(window_number_, panel_number=16)
                print(window_name)
                for j in range(16):
                    cnt += 1
                    if cnt >= N:
                        return cnt, window_number_
                    cls.run_command(session_name=session_name, window_name=window_name, panel_number=j, command_line=data_ls[cnt])
                    print(window_name, data_ls[cnt])
            return cnt, window_number_

        def run_one_window(data_ls, cnt, window_number_, panel_number):
            window_number_, window_name = create_window(window_number_, panel_number=panel_number)
            print(window_name)
            for i in range(panel_number):
                cnt += 1
                if cnt >= N:
                    return cnt, window_number_
                cls.run_command(session_name=session_name, window_name=window_name, panel_number=i, command_line=data_ls[cnt])
                print(window_name, data_ls[cnt])
            return cnt, window_number_

        if N > 16:
            ind, window_number = run_16(task_ls, cnt=ind, window_number_=window_number)
        rest_number = N - ind - 1
        if rest_number > 8:
            ind, window_number = run_one_window(task_ls, cnt=ind, window_number_=window_number, panel_number=16)
        elif rest_number > 4:
            ind, window_number = run_one_window(task_ls, cnt=ind, window_number_=window_number, panel_number=8)
        elif rest_number > 2:
            ind, window_number = run_one_window(task_ls, cnt=ind, window_number_=window_number, panel_number=4)
        elif rest_number > 0:
            ind, window_number = run_one_window(task_ls, cnt=ind, window_number_=window_number, panel_number=2)
        else:
            pass


class Enumerate_params_dict():
    """
    get several param_pools get indexes
    pool_ls: list of param_list, e.g:[[0.1,0.01,0.001],[1e-4,1e-5,1e-6]] for lr and wd pairs
    task_thread: all the experiments are divided into several threads to do

    主要用在刷参数上面，我有很多组参数学习率，weight decay，我全部的组合都要试一遍，然后呢，我又只有几张卡可以分配，于是，我就用这个类来打理
    """

    def __init__(self, task_thread: int, if_single_id_task=False, **kwarg):
        def match_pairs(pair_ls, item_key):
            item_pool = kwarg[item_key]
            if type(item_pool) != list:
                raise ValueError('wrong type of the item pool:%s' % type(item_pool))
            if len(item_pool) == 0:
                raise ValueError('item_pool should not be temp, length of the item_pool: %s' % len(item_pool))
            new_data_ls = []
            for pair in pair_ls:
                for i in range(len(item_pool)):
                    new_data_ls.append(pair + ((item_key, i),))
            return new_data_ls

        def divid_threads(ind_pairs, N: int):
            def get_param_dict(pair):
                temp = {}
                for p_key, ind in pair:
                    temp[p_key] = kwarg[p_key][ind]
                return temp

            data_ls = []
            for i in range(N):
                data_ls.append([])
            cnt = 0
            for pair in ind_pairs:
                param_dict = get_param_dict(pair)
                data_ls[cnt % len(data_ls)].append(param_dict)
                cnt += 1
            return data_ls

        data_ls = [(), ]
        item_key_ls = list(kwarg.keys())
        # print('before sort', item_key_ls)
        item_key_ls.sort()
        # print('after sort', item_key_ls)
        for item_key in item_key_ls:
            data_ls = match_pairs(data_ls, item_key)
        self.item_pool_dict = kwarg
        self.key_ind_pairs = data_ls
        if if_single_id_task:
            task_thread = len(data_ls)
        self.thread_pool = divid_threads(self.key_ind_pairs, task_thread)
        self.if_single_id_task = if_single_id_task

    def get_thread(self, ind: int):
        thread_pool = self.thread_pool[ind % len(self.thread_pool)]
        return thread_pool

    def get_pool(self):
        param_pool = self.thread_pool
        if self.if_single_id_task:
            data = []
            for i in param_pool:
                data.append(i[0])
            return data
        else:
            return param_pool

    @classmethod
    def demo(cls):
        test_id = 0
        param_pool_dict = {
            'a': [0, 1, 2],
            'b': [0, 1, 2],
        }
        task_manager = Enumerate_params_dict(task_thread=2, if_single_id_task=True, **param_pool_dict)
        param_pool = task_manager.get_thread(ind=test_id)
        print(param_pool)
        print('thread_pool', len(task_manager.thread_pool), task_manager.thread_pool)


class Main_manager():
    def __init__(self, **kwargs):
        self.mod = 'demo'  # which module to run: RAFT or others
        self.stage = 'demo'  # which stage, e.g. chairs or sintel or KITTI
        self.exid = 0  # the experiment id
        self.subexid = 0  # sub task id in this experiment
        self.check = False  # only check result
        self.runTmux = False  # do experiments
        self.multiid = -1  # used when if_multi_task_in_one > 0
        self.Env_dict = {}
        self.root_dir = ''
        self.main_tabel_key = None  # key to print result tabel

        self.demo_debug = False
        self.update(kwargs)
        self.ex_dir = ''  # dir to save results

        # params need to be defined in each experiment
        self.ex_name = 'demo'  # which experiment function to run
        self.tmux_session = 's'  # tmux session name, Note that you should open this session before the experiments
        self.tmux_name = 'demo'  # tmux window name
        self.python_env = 'base'  # python env
        self.log_name_ls = []
        self.memory = 30000
        self.cpu = 5
        self.gpu = 2
        self.num_sub_ex = 9  # number of sub experiments
        self.param_pool_dict = {}
        self.if_preemtible = False
        self.if_multi_task_in_one = -1  # run multiple task in one rlaunch, # <0 means False (one ex in one rlaunch), >=0 means True

    def to_dict(self):
        def dict_class(obj):
            temp = {}
            k = dir(obj)
            for name in k:
                if not name.startswith('_') and name not in ['log_name_dict', ]:
                    value = getattr(obj, name)
                    if callable(value):
                        pass
                    else:
                        temp[name] = value
            return temp

        s_dict = dict_class(self)
        return s_dict

    def update(self, data: dict):


        s_dict = self.to_dict()
        k_list = list(s_dict.keys())
        t_key = list(data.keys())
        for i in k_list:
            if i in t_key:
                setattr(self, i, data[i])
                # print('set param ====  %s:   %s' % (i, data[i]))

    # init experiment dir
    def init_ex_dir(self):
        # return experiment dir
        ex_dir = os.path.join(self.root_dir, self.mod, self.stage, 'ex_%s_%s' % (self.exid, self.ex_name))
        if self.demo_debug:
            print('do experiments at: %s' % ex_dir)
        else:
            self._check_dir(ex_dir)
        self.ex_dir = ex_dir

    # init sub experiment dir
    def init_subex_dir(self):
        # return experiment dir
        ex_dir = os.path.join(self.ex_dir, 'subex_%s' % self.subexid)
        if self.demo_debug:
            print('do this subex at: %s' % ex_dir)
        else:
            print('do this subex at: %s' % ex_dir)
            self._check_dir(ex_dir)
        self.ex_dir = ex_dir

    def _prepare_log_name(self, params):
        log_print = '(%s,%s),' % (self.exid, self.subexid)
        for i in self.log_name_ls:
            temp_value = params[i]
            if type(temp_value) == str:
                show_value = temp_value
            else:
                # show_value = '-'.join([str(tempi) for tempi in temp_value])
                show_value = '%s%s' % (temp_value, '')
            if i in self.log_name_dict.keys():
                i_name = self.log_name_dict[i]
            else:
                i_name = i
            log_print += '%s[%s],' % (i_name, show_value)
        return log_print

    def _get_sub_ex_ids(self):
        a = list(range(self.num_sub_ex))
        return a

    def _run_tmux_command_lines(self):
        def _divede_multi_task(task_ls_, n_task):
            data_ls_ = []
            for i in range(n_task):
                data_ls_.append([])
            for ind, k in enumerate(task_ls_):
                data_ls_[ind % len(data_ls_)].append(k)
            return data_ls_

        if self.if_preemtible:
            rlaunch_str = 'rlaunch --memory=%s --cpu=%s --gpu=%s --preemptible=best-effort -- ' % (self.memory, self.cpu, self.gpu)
        else:
            rlaunch_str = 'rlaunch --memory=%s --cpu=%s --gpu=%s -- ' % (self.memory, self.cpu, self.gpu)
        if self.demo_debug:
            rlaunch_str = ''
        run_python_str = '%s main.py' % self.Env_dict[self.python_env]
        su_id_ls = self._get_sub_ex_ids()
        task_no_rlaunch_ls = ['%s --mod %s --stage %s --exid %s --check %s --runTmux %s --subexid %s' %
                              (run_python_str, self.mod, self.stage, self.exid, False, False, i) for i in su_id_ls]
        self.tmux_name += '%s' % self.exid

        if self.if_multi_task_in_one >= 0:
            # 如果要一个窗口跑多个任务，那就先开窗口, 出rlaunch，然后指定multi这个指令来分跑任务
            multi_task_ls = _divede_multi_task(task_ls_=task_no_rlaunch_ls, n_task=self.if_multi_task_in_one)  # 把所有要跑的任务分成多份，每份在一个window跑
            # if self.demo_debug:
            #     print('=== do multi task in one window: multiid=%s' % self.multiid)
            if self.multiid == -1:  # 没有给定multi id的时候为-1，这个时候要开窗口，每个窗口给一个multi id去分跑任务，注意runTmux=True
                # if self.demo_debug:
                #     print('=== prepare multi task: multiid=%s' % self.multiid)
                multi_window_ls = ['%s%s --mod %s --stage %s --exid %s --multiid %s --check %s --runTmux %s --subexid %s' %
                                   (rlaunch_str, run_python_str, self.mod, self.stage, self.exid, i, False, True, -1) for i in range(self.if_multi_task_in_one)]
                Tmux_line.run_task(task_ls=multi_window_ls, task_name=self.tmux_name, session_name=self.tmux_session)
            else:  # 指定了multi id了, 那就分跑就好了，注意runTmux=True
                task_ls = multi_task_ls[self.multiid % len(multi_task_ls)]
                # if self.demo_debug:
                #     print('=== run multi task: multiid=%s, runTmux=%s' % (self.multiid, self.runTmux))
                for task_command in task_ls:
                    # print(task_command)
                    # if self.demo_debug:
                    #     print('muti-task %s' % self.multiid, str(task_command))
                    # else:
                    #     os.system(str(task_command))
                    os.system('%s%s' % (rlaunch_str, task_command))
        else:  # 注意，注意subexid不等于-1，跑对应的任务
            task_ls = ['%s%s' % (rlaunch_str, i) for i in task_no_rlaunch_ls]
            # if self.demo_debug:
            #     for i in task_ls:
            #         print(i)
            Tmux_line.run_task(task_ls=task_ls, task_name=self.tmux_name, session_name=self.tmux_session)

    # make experiment parameters
    def _prepare_param_pool(self):
        # if_single_id_task=True, do not change this
        task_manager = Enumerate_params_dict(task_thread=0, if_single_id_task=True, **self.param_pool_dict)
        params_pool = task_manager.get_pool()
        # print(params)
        return params_pool

    def _prepare_result_tabel_dict(self):
        param_pool = self._prepare_param_pool()  # param_pool: list contains many dict, [{training param1},{training params2},...]
        ex_ids = self._get_sub_ex_ids()

        # get all keys in param pool dict we are going to compare
        temp_keys = list(self.param_pool_dict.keys())
        temp_keys = sorted(temp_keys, reverse=False)

        tabel_key_ls = []  # get all keys with multiple value
        for i in temp_keys:
            temp_n = len(self.param_pool_dict[i])
            if temp_n > 1:
                tabel_key_ls.append(i)

        # check main key
        if self.main_tabel_key is None:
            if len(tabel_key_ls) < 1:
                self.main_tabel_key = temp_keys[0]
                tabel_key_ls.append(self.main_tabel_key)
            else:
                self.main_tabel_key = tabel_key_ls[0]
        elif self.main_tabel_key in tabel_key_ls:  # maybe
            pass
        else:
            tabel_key_ls.append(self.main_tabel_key)
        assert self.main_tabel_key in temp_keys  # check the main_tabel_key is right

        # === match id to its key value
        data = {}
        for i in ex_ids:
            param = param_pool[i % len(param_pool)]
            data[i] = {k: param[k] for k in tabel_key_ls}
        return data

    def check_results(self, param_pool):
        # param_pool: {'id':{'key1':value1,'key2':value2,...},...}
        '''
        I want to get:
        {
            'mainKeyValue1':[{'id','filedir','score','valinfo','key1','key2',...},...], # soreted this list by scores, the lowest is in the end
            mainKeyValue2':[{'id','filedir','score','valinfo','key1','key2',...},...],
            ...
        }
        '''
        from prettytable import PrettyTable

        def get_subex_id_ls(temp, r_dir):
            d_ = {}
            for t in temp:
                try:
                    subid = t.split('_')[1]
                    subid = int(subid)
                except:
                    raise ValueError('fail to fetch sub ex id: %s' % t)
                if subid in d_.keys():
                    raise ValueError('multiple ex for %s and %s' % (t, d_[subid]))
                d_[subid] = os.path.join(r_dir, t)
            return d_

        def get_tabel_key_ls(param_pool_, main_tk):
            t_ks = list(param_pool_[list(param_pool_.keys())[0]].keys())
            t = [main_tk, ]
            for tk in t_ks:
                if tk != main_tk:
                    t.append(tk)
            return t

        if len(param_pool.keys()) < 1:
            print('no param_pool  %s%s' % (param_pool, ''))
            return
        assert self.main_tabel_key is not None

        print('to do: check results from %s' % self.ex_dir)
        tabel_keys = get_tabel_key_ls(param_pool, self.main_tabel_key)  # the order is ok
        tabel_keys += ['ex_id', 'score', 'epoch', 'valinfo', 'ex_dir']
        if self.demo_debug:
            for temp_i in param_pool.keys():
                print('%s: %s' % (temp_i, param_pool[temp_i]))
            ex_id_d = {k: 'ex_%s' % k for k in param_pool.keys()}
        else:
            files = os.listdir(self.ex_dir)
            ex_id_d = get_subex_id_ls(files, self.ex_dir)  # {'0':dir_name,'1':dir_name,...}

        tb_dict = {}
        # make the tabel dict i want
        for ind in ex_id_d.keys():
            data = param_pool[ind]
            m_value = '%s%s' % (data[self.main_tabel_key], '')
            if m_value not in tb_dict.keys():
                tb_dict[m_value] = []
            score, epoch, valinfo = self.get_result_info(ex_id_d[ind])
            if score is None:
                continue
            ind_res = {'ex_id': ind, 'ex_dir': ex_id_d[ind], 'score': score, 'epoch': epoch, 'valinfo': valinfo}
            ind_res.update(param_pool[ind])
            tb_dict[m_value].append(ind_res)

            # print(info)

        # print tabel
        if self.demo_debug:
            for temp_i in tb_dict.keys():
                print('\n=== %s' % temp_i)
                for temp_j in tb_dict[temp_i]:
                    print(temp_j)

        # sort and print the tabel
        tb = PrettyTable(field_names=tabel_keys)
        tb.align[self.main_tabel_key] = "l"  # 以main_tabel_key字段左对齐
        tb.padding_width = 1  # 填充宽度
        for mv in tb_dict.keys():
            mv_lines = tb_dict[mv]
            mv_lines = sorted(mv_lines, key=lambda x: x['score'], reverse=True)
            # for i in mv_lines:
            #     print(type(i['score']), i['score'])
            for data_line in mv_lines:
                data_line['score'] = '%.3f' % data_line['score']
                tb.add_row(['%s%s' % (data_line[tbk], '') for tbk in tabel_keys])
            tb.add_row(['%s%s' % ('-' * len(tbk), '') for tbk in tabel_keys])
            print('  ')
        print(tb)

    def run(self):
        '''
        --从main函数拿到参数：1）选择哪个模块；2）选择哪个实验类；3）实验id是多少；4）是否只是check实验结果；5）是否用tmux大批量跑实验；6）批量实验id
        ----这里根据1)模块，2)实验类，3)实验id得到实验类(此class)，
        ----首先确认实验信息('define_experiments')：参数池（可排列组合匹配出多套实验参数），子实验个数，其他参数：实验memory，cpu，gpu个数,虚拟环境名, tmux的window名(tmux_name)等
        ----初始化实验路径('init_ex_dir')
        ------是否check实验结果
        --------是（进行实验结果汇总）
        ----------实验路径不存在-->无实验结果
        ----------实验路径存在--->调用'check_results'函数，进行搜索实验结果并打表
        --------否（要进行实验），开始进进出出安排实验
        ----------判断是否用tmux大批量跑实验(runTmux)：
        ------------是：根据子实验个数，安排好实验命令，批量开tmux运行起来: '_run_tmux_command_lines'
        ------------否：说明上一步已经完成，现在要开始运行实验，
        --------------根据参数池和子实验个数排列组合出多套实验参数: '_prepare_param_pool'
        --------------根据6）批量实验id(subexid)选择此实验的参数: 'param'
        --------------将实验参数传给相关模块的相关函数开始实验: 'do_training(param)'
        '''
        self.define_experiments()
        self.init_ex_dir()
        # === run
        if self.check:  # check experiment results and print table
            # make param pool
            self.check_results(self._prepare_result_tabel_dict())
        else:  # do experiments
            if self.runTmux:  # open tmux and command lines
                self._run_tmux_command_lines()
            else:
                # make param pool
                param_pool = self._prepare_param_pool()  # param_pool: list contains many dict, [{training param1},{training params2},...]
                param = param_pool[self.subexid % len(param_pool)]  # param for this sub ex task
                self.init_subex_dir()
                self.do_training(param)

    @classmethod
    def _check_dir(cls, path):
        if not os.path.exists(path):
            os.makedirs(path)

    @classmethod
    def check_argparse(cls):
        def check_bool(k, v):
            if v == 'True':
                return True
            if v == 'False':
                return False
            raise ValueError('''wrong value: %s for input parameter: %s, it should be string of 'True' or 'False' ''' % (v, k))

        parser = argparse.ArgumentParser()
        parser.add_argument('-m', '--mod', type=str, default='demo')
        parser.add_argument('-s', '--stage', type=str, default='chairs')
        parser.add_argument('-e', '--exid', type=int, default=-1)
        parser.add_argument('-c', '--check', type=str, default='True')  # False,True
        # only care about params above
        parser.add_argument('-si', '--subexid', type=int, default=-1)
        parser.add_argument('-mi', '--multiid', type=int, default=-1)
        parser.add_argument('-rt', '--runTmux', type=str, default='True')
        # problme in python argparse transform bool parameter: https://blog.csdn.net/a15561415881/article/details/106088831
        args = parser.parse_args()
        # print(args)
        # print('====')
        arg_d = vars(args)
        arg_d['runTmux'] = check_bool('runTmux', arg_d['runTmux'])
        arg_d['check'] = check_bool('check', arg_d['check'])
        return arg_d

    # ===================Note: functions below should be redefined in your own experiments====================================
    def do_training(self, param):
        log_name = self._prepare_log_name(param)
        print('=== do_training')
        print('to do ex, log_name=  %s' % log_name)
        print('parameter=: %s' % param)

    # TODO check experiment results from experiment dir
    def get_result_info(self, sub_ex_dir: str):
        score, epoch, valinfo = 0, 0, 0
        return score, epoch, valinfo

    @property
    def log_name_dict(self):
        # prepare params in training log
        data = {
            'weight_decay': 'wd',
            'lr': 'lr'
        }
        return data

    # init parameter pool to do experiments
    def define_experiments(self):
        self.tmux_session = 's'  # tmux session name, Note that you should open this session before the experiments
        self.tmux_name = 'demo'  # tmux window name
        self.python_env = 'no'  # python env
        self.memory = 30000
        self.cpu = 5
        self.gpu = 2
        self.if_preemtible = True

        if self.exid == -1:
            self.param_pool_dict = {
                'lr': [0.1, 0.2, ],
                'weight_decay': [0, 1, 2],
            }
            self.num_sub_ex = 9
            self.ex_name = 'demo_ex'  # name of this experiment(also used as result dir name)
            self.log_name_ls = ['lr', 'weight_decay']  # which param to show in the training log
            self.if_multi_task_in_one = -1  # <0 means False (one ex in one rlaunch), >=0 means True, because 'range' begins from 0
            self.main_tabel_key = 'weight_decay'
        else:
            raise ValueError('experimenr not implemented: id =%s ' % self.exid)
