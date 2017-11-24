import tensorflow as tf
import numpy as np
import os
import time
import itertools
from model import Bi_lstm
from Datahelpers import Datahelper
import datetime

tf.app.flags.DEFINE_integer("batch_size", 1, "Batch size of train set.")
tf.app.flags.DEFINE_string("cell", "lstm", "Rnn cell.")
tf.app.flags.DEFINE_integer("max_epoch", 15, "Number of training epoch.")
tf.app.flags.DEFINE_integer("hidden_size", 300, "Size of each layer.")
tf.app.flags.DEFINE_integer("word_emb_size", 50, "Size of word embedding.")
tf.app.flags.DEFINE_integer("pos_emb_size", 20, "Size of embedding.")
tf.app.flags.DEFINE_integer("feat_emb_size", 5, "Size of embedding.")
tf.app.flags.DEFINE_integer("limits", 0,'max data set size')
tf.app.flags.DEFINE_string("gpu", '0', "GPU id.")
tf.app.flags.DEFINE_string("dir",'./data','data set directory')
tf.app.flags.DEFINE_string("mode",'test','train or test')
tf.app.flags.DEFINE_integer("report", 2000,'report')
tf.app.flags.DEFINE_string("save",'file_3o','save directory')
tf.app.flags.DEFINE_string("word_emb",'sskip','pretrained_embedding')
tf.app.flags.DEFINE_string("rnn_save",'rnn_saved','save directory')
tf.app.flags.DEFINE_string("pretrained",'False','save directory')
tf.app.flags.DEFINE_string("use_emb",'False','whether use pretrained embeddings')
FLAGS = tf.app.flags.FLAGS
# 89.96
if FLAGS.save == "file_3o":
    save_dir = './' + FLAGS.save + '/'
    #os.mkdir(save_dir)
else:
    save_dir = FLAGS.dir + '/' + FLAGS.rnn_save + '/'

def convert(tags):
    for i, sent in enumerate(tags):
        for j, cur_tag in enumerate(sent):
            tag = cur_tag.split('-')
            if len(tag) == 2:
                if tag[0] == 'S':
                    cur_tag = 'B-'+ tag[1]
                    tags[i][j] = cur_tag
                elif tag[0] == 'E':
                    cur_tag = 'I-' + tag[1]
                    tags[i][j] = cur_tag
    return tags


model_dir = './data/saved_model/'
log_file = './log.txt'
with open(FLAGS.dir +"/ned.testa") as infile:
    gold_dev = [[w for w in sent.strip().split('\n')]for sent in infile.read().split('\n\n')]
with open(FLAGS.dir +"/ned.testb") as infile_test:
    gold_test = [[w for w in sent.strip().split('\n')]for sent in infile_test.read().split('\n\n')]


with open("./data/outfile_test_1o.txt") as prob_file_1o:
    test_sent_1o = [[[float(w) for w in sent.split(' ') if w != ''] for sent in sentence.split('\n')]for sentence in prob_file_1o.read().split('\n\n')]
with open("./data/outfile_test_2o.txt") as prob_file_2o:
    test_sent_2o = [[[float(w2) for w2 in sent2.split(' ') if w2 != ''] for sent2 in sentence2.split('\n')]for sentence2 in prob_file_2o.read().split('\n\n')]
with open("./data/tagfile_test_1o.txt") as tag_file_test:
    test_all_tag = [[[int(t) for t in word.strip().split(' ')] for word in sentence.strip().split('\n')] for sentence in
                    tag_file_test.read().strip().split('\n\n')]

with open("./data/outfile_dev_1o.txt") as prob_file:
    dev_sent_1o = [[[float(w) for w in sent.split(' ') if w != ''] for sent in sentence.split('\n')]for sentence in prob_file.read().split('\n\n')]
with open("./data/outfile_dev_2o.txt") as prob_file2:
    dev_sent_2o = [[[float(w2) for w2 in sent2.split(' ') if w2 != ''] for sent2 in sentence2.split('\n')]for sentence2 in prob_file2.read().split('\n\n')]
with open("./data/tagfile_dev_1o.txt") as tag_file:
    dev_all_tag = [[[int(t) for t in word.strip().split(' ')] for word in sentence.strip().split('\n')] for sentence in
               tag_file.read().strip().split('\n\n')]



def write_log(s):
    with open(log_file, 'a') as f:
        f.write(s)

def train(sess, datahelper, model):
    if FLAGS.pretrained == 'True':
        model.load(sess,save_dir)
        evaluate(sess, datahelper, model, gold_dev)

    write_log("##############################\n")
    for flag in FLAGS.__flags:
        write_log(flag + " = " + str(FLAGS.__flags[flag]) + '\n')
    write_log("##############################\n")
    train_set = datahelper.train_set
    global_step = 0
    best_result = 80.
    for _ in range(FLAGS.max_epoch):
        loss, start_time = 0.0, time.time()
        for x in datahelper.batch_iter(train_set, FLAGS.batch_size, True, datahelper.feat_num):
            loss += model(sess, x)
            global_step += 1
            if (global_step % FLAGS.report == 0):
                cost_time = time.time() - start_time

                write_log("%d : loss = %.3f, time = %.3f \n" % (global_step // FLAGS.report, loss, cost_time))
                print ("%d : loss = %.3f, time = %.3f " % (global_step // FLAGS.report, loss, cost_time))
                loss, start_time = 0.0, time.time()
                result_dev = evaluate(sess, datahelper, model, gold_dev, 'dev')
                result_test = evaluate(sess, datahelper, model, gold_test, 'test')
                F1_dev,  overall_result_dev = result_dev
                F1_test, overall_result_test = result_test

                if F1_dev > best_result:

                    print ("saving model......")
                    cur_save_dir = './file_3o/' + 'git_dev'+ str(F1_dev) + '_' + 'test'+ str(F1_test) + '/'
                    os.mkdir(cur_save_dir)
                    model.save(sess, cur_save_dir)
                    print ("model with " + 'dev'+str(F1_dev) +'_test' + str(F1_test) + ' saved')
                    best_result = F1_dev

def test(sess, datahelper, model, save_dir):
    #save_dir = './file_3o/dev75.26_test76.47/'
    model.load(sess, save_dir)
    print ("save_dir = ",save_dir)
    evaluate(sess,datahelper, model, gold_test, 'test')

def evaluate(sess, datahelper, model, gold, data_set):
    if data_set == 'dev':
        test_set = datahelper.dev_set
        sent_1o = dev_sent_1o
        sent_2o = dev_sent_2o
        ktag_1o = dev_all_tag
    elif data_set == 'test':
        test_set = datahelper.test_set
        sent_1o = test_sent_1o
        sent_2o = test_sent_2o
        ktag_1o = test_all_tag
    else:
        pass

    pred = []
    i = 0
    k = 5 #top5 order-1 tags
    
    # multi-order-decoding
    for x in datahelper.batch_iter(test_set, FLAGS.batch_size, False, datahelper.feat_num):
        predictions, prob = model.generate(sess, x)
        for idx_in_batch in range(FLAGS.batch_size):
            sent_length = x['x_len'][idx_in_batch]
            beta_1o = sent_1o[i * FLAGS.batch_size + idx_in_batch]
            beta_2o = sent_2o[i * FLAGS.batch_size + idx_in_batch]
            beta_3o = prob[idx_in_batch]

            k_tag = ktag_1o[i * FLAGS.batch_size + idx_in_batch]
            path = [[[-1 for n in range(len(datahelper.idx_1o_tag))] for m in range(len(datahelper.idx_1o_tag))] for p in range(sent_length)]
            dp = [[[-9999. for n in range(len(datahelper.idx_1o_tag))] for m in range(len(datahelper.idx_1o_tag))] for p in range(sent_length)]

            pre = datahelper.tag_1o_idx['PADDING']
            for cur in range(len(datahelper.idx_1o_tag)):
                for nex in range(len(datahelper.idx_1o_tag)):
                    if (pre, cur, nex) in datahelper.tag_3o_idx:
                        idx_3o = datahelper.tag_3o_idx[pre, cur, nex]
                        idx_2o = datahelper.tag_2o_idx[pre, cur]
                        dp[0][cur][nex] = np.log(beta_3o[0][idx_3o]) + np.log(beta_2o[0][idx_2o]) + np.log(
                            beta_1o[0][cur])
                        path[0][cur][nex] = idx_3o

            for p in range(1, sent_length - 1):
                for pre_tag in range(k):
                    pre = k_tag[p - 1][pre_tag]
                    for cur_tag in range(k):
                        cur = k_tag[p][cur_tag]
                        for nex_tag in range(k):
                            nex = k_tag[p + 1][nex_tag]
                            if (pre, cur, nex) in datahelper.tag_3o_idx:
                                idx_2o = datahelper.tag_2o_idx[pre, cur]
                                idx_3o = datahelper.tag_3o_idx[pre, cur, nex]
                                temp = dp[p - 1][pre][cur] + np.log(beta_3o[p][idx_3o]) + np.log(
                                    beta_2o[p][idx_2o]) + np.log(
                                    beta_1o[p][cur])
                                if path[p][cur][nex] == -1 or dp[p][cur][nex] < temp:
                                    dp[p][cur][nex] = temp
                                    path[p][cur][nex] = idx_3o
            p = sent_length - 1
            for pre_tag in range(k):
                pre = k_tag[p - 1][pre_tag]
                for cur_tag in range(k):
                    cur = k_tag[p][cur_tag]
                    nex = datahelper.tag_1o_idx['PADDING']
                    if (pre, cur, nex) in datahelper.tag_3o_idx:
                        idx_2o = datahelper.tag_2o_idx[pre, cur]
                        idx_3o = datahelper.tag_3o_idx[pre, cur, nex]
                        temp = dp[p - 1][pre][cur] + np.log(beta_3o[p][idx_3o]) + np.log(beta_2o[p][idx_2o]) + np.log(
                            beta_1o[p][cur])
                        if path[p][cur][nex] == -1 or dp[p][cur][nex] < temp:
                            dp[p][cur][nex] = temp
                            path[p][cur][nex] = idx_3o

            tag = []
            pos = sent_length - 1
            (j_3, k_3) = max(
                [(dp[pos][x][y], (x, y)) for x, y in itertools.product(range(len(datahelper.tag_1o_idx)), repeat=2)
                 if
                 path[pos][x][y] != -1])[1]

            while pos >= 0:
                tag.insert(0, datahelper.idx_3o_tag[path[pos][j_3][k_3]][1])
                _p = path[pos][j_3][k_3]
                j_3 = datahelper.idx_3o_tag[_p][0]
                k_3 = datahelper.idx_3o_tag[_p][1]
                pos -= 1

            pred.append(tag)
        i += 1

    pred_tags = []
    for pred_sent in pred:
        sent_tags = []
        for tag in pred_sent:
            sent_tags.append(datahelper.idx_1o_tag[tag])
        pred_tags.append(sent_tags)
    pred = convert(pred_tags)

    with open("outfile_%s.txt"%data_set, 'w') as f:
        for test_sent, pred_sent in zip(gold, pred):
            pre_tag = 'NULL'
            for test_line, pred_line in zip(test_sent, pred_sent):
                test_line = test_line.strip().split()
                sp_pretag = pre_tag.split('-')
                cur_tag = pred_line
                sp_curtag = cur_tag.split('-')

                if len(sp_pretag) == 2 and len(sp_curtag) == 2:
                    pre_chunk = sp_pretag[0]
                    pre_type = sp_pretag[1]
                    cur_chunk = sp_curtag[0]
                    cur_type = sp_curtag[1]
                    if pre_chunk == 'B' and cur_chunk == 'I' and pre_type != cur_type:
                        cur_tag = cur_chunk + '-' + pre_type

                pre_tag = cur_tag
                test_line.append(cur_tag)
                f.write('{}\n'.format(" ".join(test_line)))
            f.write("\n")
    exe_command = 'perl conlleval < outfile_%s.txt'%data_set
    result = os.popen(exe_command).readlines()
    for line in result:
        write_log(line)
    F1 = (result[1].split('  '))[-1]
    print ("%s F1 score = %s" %(data_set, F1))
    return float(F1), result[1]

def main():
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        datahelper = Datahelper(FLAGS.dir, FLAGS.limits)
        vocab_size = len(datahelper.word2idx) + 2
        pos_size = len(datahelper.pos2idx)
        tag_size = len(datahelper.tag_3o_idx)
        print (datahelper.tag_1o_idx)
        print ("size of 3o tagset = ", len(datahelper.idx_3o_tag))
        feat_size = len(datahelper.feat2idx)
        model = Bi_lstm(FLAGS.batch_size, vocab_size, pos_size,  FLAGS.word_emb_size,
                        FLAGS.pos_emb_size, FLAGS.hidden_size, tag_size,
                        FLAGS.use_emb, feat_size, datahelper.feat_num, FLAGS.feat_emb_size)
        sess.run(tf.global_variables_initializer())
        if FLAGS.mode == 'train':
            train(sess, datahelper, model)
        if FLAGS.mode == 'test':
            test(sess, datahelper, model, model_dir)

if __name__ == '__main__':
    with tf.device('/gpu:' + FLAGS.gpu):
        main()
