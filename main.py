# 音乐与数学——遗传算法作曲
# 作者：郝炜天, 陈鹏
# 2023-5

import random
import numpy as np
import matplotlib.pyplot as plt

#钢琴音色对应B版本pysynth
import pysynth_b as ps

##----------------------------格式转换模块---------------------

## pysynth乐谱说明
#一种序列表示法为二维列表[['c',4],['b5',2]...]
#列表中每个元素为二元列表，对应一个含时值的音。第一个元素完整表示是'c#5*',其中'c'对应音名，'#'升降符，'5'为第五个八度，默认'4','*'表示重音。

## 音名索引
# 'a' ~ 'f'的音名序列
a2f = [chr(ord('a')+i) for i in range(7)]
# 'c' ~ 'b' 的音名序列
c2b = np.roll(a2f,-2).tolist()
# 0~14 对应的音名，默认c4级的不带4，c5带
note2abc = ['r'] + c2b + [note + '5' for note  in c2b]

## 时值索引（含‘15’的个数.这里基准是4分音符，最多4个一小节对应'1',负号对应一个附点。若要更复杂的需要通读pysynth了解完全的时值表示）,并仅修改temp列表即可拓展时值表示。
temp = [8,4,-4,2,2,-2,-2,1] #8分音符为基准。4,6个'15'不知道怎么处理

#格式转换1： 简谱notation转化为音名谱syn.不考虑升降号及重音，且不带附点)
def note2syn(notation):
    syn = []
    for note in notation:
        if note != 15:
            syn.append([note2abc[note],temp[0]])#默认0个15
        else:
            num15 = temp.index(syn[-1][1])+1
            syn[-1][1] = temp[num15]
    return syn

#格式转换2： 简谱notation转化为音名谱syn.不考虑升降号及重音，时值直接由含15的个数+1表示)
def note2syn_time(notation):
    syn = []
    for note in notation:
        if note != 15:
            syn.append([note2abc[note],1])
        else:
            syn[-1][1] += 1
    return syn
#格式转换3： syn转化回简谱
def syn_time2note(syn_time):
    notation = []
    for note,time in syn_time:
        notation.append(note2abc.index(note))
        notation += [15]*(time-1)
    return notation

##----------------------------适应度函数分量模块---------------------

## 和谐度评分

#和谐度评分表（按音程差）：
pitch_harm = {
    1:5,
    8:5,#纯一度、纯八度
    4:4,
    5:4,#纯四、五
    3:3,#大小三
    6:2,#大小六
    2:1,
    7:1#大小二、七
}

#两个单音间的和谐度
def note_harm(note1,note2):
    pitch_diff = ( note2abc.index(note1) - note2abc.index(note2) )%7 + 1 #互补的音程关系某种意义上等价；或说如果设置互补音程谐和度不同，则八度音符不等价
    return pitch_harm[pitch_diff]

#一个单音列与和弦的平均谐和度
def chord_harm(chord,individual):#输入的individual是pysynth中的二元表示法;chord仅由音名组成
    score = 0
    max_score = 0
    min_score = 0
    for note_c in chord:
        for note_i,temp in individual:
            if note_i[0] != 'r': #暂不考虑休止符
                time = temp
                #dot = 1 if temp > 0 else -3/2
                #time = dot/temp #pysynth, 时值作为权重,temp中的数字比如4，代表4分音符；负号表示附点，使得时长变为原来的3/2
                score += note_harm(note_c,note_i)*time
                max_score += 5*time
                min_score += time
    if max_score == 0:
        max_score = 1
    return (score-min_score)/max_score #归一化

#和弦识别
chord_sum = 7
chords = [np.array(c2b)[[i%7,(i+2)%7,(i+4)%7]] for i in range(chord_sum)]#七级三和弦
def chord_cognize(syn):
    #和弦估计
    chord_evl = [chord_harm(chord,syn) for chord in chords]
    max_chord_evl = max(chord_evl)
    chord_level = chord_evl.index(max_chord_evl)
    return chord_level, max_chord_evl
refer_chord_evolution = ['c','f','e','a','f','g','c','c']
refer_chord_evolution_level = np.array([c2b.index(refer) for refer in refer_chord_evolution])

chord_log = []
#和弦相关表现的估计
def chord_evl(chord_levels,max_chord_evls):
    #chord_evolution = inner_harm([[c2b[chord_level],1] for chord_level in chord_levels]) #演进要求和谐
    chord_ev_dis = np.linalg.norm(np.array(chord_levels)-refer_chord_evolution_level)
    chord_evolution = 1/(1+chord_ev_dis)*8 #norm
    chord_match = np.linalg.norm(max_chord_evls)*2
    
    chord_scores = np.array([chord_evolution,chord_match])
    chord_weight = np.array([0.4,0.5])
    chord_score = (chord_scores*chord_weight).sum()
    
    chord_log.append(chord_scores)
    return  chord_score #每小节

#小节内的旋律、节奏按八分为主的升调式
ref_melody_method = [1,1,1,0,1,1,0] #是否升调
ref_rhythm = [1,1,1,1,1,1,2]
mel_rhy_log = []
def melody_rhythm_match(syn):
    melody_dis = 0
    rhythm_dis = 0
    for i in range(1,len(syn)):
        melody_dis += (int((note2abc.index(syn[i][0])-note2abc.index(syn[i-1][0]))>=0) - ref_melody_method[i-1])**2
        rhythm_dis += (ref_rhythm[i-1]-syn[i][1])**2
    #mel_rhy = np.array([[note2abc.index(note),time] for note,time in syn])
    
    mel_rhy_scores = np.array([1/(melody_dis+1) , 1/(rhythm_dis+1)])
    mel_rhy_weight = np.array([0.5,0.5])
    mel_rhy_score = (mel_rhy_scores*mel_rhy_weight).sum()
    
    mel_rhy_log.append(mel_rhy_scores)
    return mel_rhy_score

# 结构性评分
def structure_evl(notation):
    bar_num = len(notation)//8
    chord_levels = []
    max_chord_evls = []
    mel_rhy_score = 0
    for bar_index in range(bar_num):
        bar = list(notation[bar_index*8:(bar_index+1)*8])
        while bar[0] == 15:
            bar.pop(0)
        syn = note2syn_time(bar)
        chord_level, max_chord_evl = chord_cognize(syn)
        chord_levels.append(chord_level)
        
        mel_rhy_score += melody_rhythm_match(syn)
        max_chord_evls.append(max_chord_evl)
    
    chord_score = chord_evl(chord_levels,max_chord_evls)
        
    scores = np.array([chord_score,mel_rhy_score])
    structure_weight = np.array([0.5,0.5])
    structure_score = (scores*structure_weight).sum()
    
    return structure_score

#一个单音列的内部和谐度
def inner_harm(individual):#输入的individual是pysynth中的二元表示法;
    score = 0
    max_score = 0
    min_score = 0
    for index in range(1,len(individual)):
        note_i, time = individual[index]
        note_pre, time_pre = individual[index-1]
        if note_i[0] != 'r': #暂不考虑休止符
            score += note_harm(note_pre,note_i)*time*time_pre
            max_score += 5*time*time_pre
            min_score += time*time_pre
    return (score-min_score)/max_score #归一化


#重复度的估计
def repeat_evl(notation):
    score = 0
    repeat_num = 1
    for index in range(1,len(notation)):
        note_i = notation[index]
        note_pre = notation[index-1]
        if note_i == note_pre:
            repeat_num += 1
            score += int(repeat_num>2)*3 + int(repeat_num>4)*1000 #限制三个重复；超过5个直接截断（尤其是15，0）
        else:
            repeat_num = 1
    return score/len(notation)

#参考旋律评分
def refer_evl(notation):
    #reference
    new_melody = np.array(notation)
    error = np.linalg.norm(new_melody-ref_melody)
    refer_score = 1/(error+1)
    return refer_score
whole_log = []

#整体性评分
def whole_evl(notation):
    syn = note2syn_time(notation)
        
    inner_harm_score = inner_harm(syn)
    repeat_score = repeat_evl(syn)
    refer_score = refer_evl(notation)
    
    scores = np.array([repeat_score, inner_harm_score, refer_score])
    whole_weight = np.array([-0.5,1.2,0])
    whole_score = (scores*whole_weight).sum()
    
    whole_log.append(np.array([repeat_score, inner_harm_score, refer_score]))
    return whole_score

#评分分量记录
def average_log(origin_log):
    arr = np.array(origin_log)
    return arr[:len(origin_log)//100*100].reshape((100,-1,arr.shape[-1])).mean(axis=1)

##----------------------------遗传算法框架---------------------
#适应度函数
def fitness_evaluation(mutated_population):
    fitness_value = []
    for item in mutated_population:
        #小节为单位分析
        scores = np.array([structure_evl(item),whole_evl(item)])
        fitness_weight = np.array([0.9,0.1])
        fitness = (fitness_weight*scores).sum()
        fitness_value.append(fitness)
    return fitness_value

# 定义被选择的概率，选中的概率与旋律适应度有关
def selection(input):
    fitness_value = fitness_evaluation(input)
    # 归一化，得到的就是每段旋律被抽到的概率，适应度越高越容易被抽到
    fitness_one = [item/sum(fitness_value) for item in fitness_value]
    for i in range(1, len(fitness_one)):
        fitness_one[i] += fitness_one[i-1]
        # 变异时从0到1均匀分布抽取数，抽出的数所在的区间代表需要变异的旋律
        # 例如，抽出0.45，在fitness_one[3]和fitness_one[4]之间，那么就让下标为4的旋律变异
    return fitness_one


#transposition 移调
# 将一个小节整体移动一个音，这里需要注意不能选到0和15
def transposition(input, fitness_input, n): 
    output = []
    for i in range(n):
        rand = random.uniform(0, 1)
        index = 0
        for j in range(0, len(input)):
            if fitness_input[j] > rand:
                index = j
                break
        
        tmp = input[index].copy() # 注意深拷贝
        rand_bar_index = random.randint(0, 7)
        rand_note = random.randint(1,14)
        for rand_index in range(rand_bar_index*8,(rand_bar_index+1)*8):
            note = temp[rand_index] #传引用
            if note != 0 and note != 15:
                note = rand_note
                
        output.append(tmp)
    print("transposition!")
    return output

#inversion 倒影
# 将某小节关于其首音镜面映射，单位为一度
def inversion(input, fitness_input, n): 
    output = []
    for i in range(n):
        rand = random.uniform(0, 1)
        index = 0
        for j in range(0, len(input)):
            if fitness_input[j] > rand:
                index = j
                break
        
        tmp = input[index].copy() # 注意深拷贝
        rand_bar_index = random.randint(0, 7)
        first_note = False
        for rand_index in range(rand_bar_index*8,(rand_bar_index+1)*8):
            note = tmp[rand_index]
            if note != 0 and note != 15:
                if not first_note:
                    first_note = note
                else:# 倒影c=2a-b, 但是要考虑高两个八度等价 -1 %14 (c-1)=[2(a-1)-(b-1)]%14
                    note = (2*first_note - note -1)%14 + 1
                
        output.append(tmp)
    print("inversion!")
    return output

# retrograde 逆行变换
# 将任意长度的某小段时序颠倒
def retrograde(input, fitness_input, n): 
    output = []
    for i in range(n):
        rand = random.uniform(0, 1)
        index = 0
        for j in range(0, len(input)):
            if fitness_input[j] > rand:
                index = j
                break
        
        tmp = reference_melody.copy() # 注意深拷贝
        syn = note2syn_time(tmp)
        start = random.randint(0,len(syn)-1)
        end = random.randint(start+1,len(syn)) 
        syn[start:end] = reversed(syn[start:end])
        tmp = syn_time2note(syn)
            
        output.append(tmp)
    print("retrograde!")
    return output


#Crossover 交叉
def crossover(input, fitness_input, n):
    output = []
    for i in range(n):
        # 首先选取两段旋律，注意不能抽到一样的
        rand = random.uniform(0, 1)
        index1 = 0
        for j in range(0, len(input)):
            if fitness_input[j] > rand:
                index1 = j
                break
        rand = random.uniform(0, 1)
        index2 = 0
        for j in range(0, len(input)):
            if fitness_input[j] > rand:
                index2 = j
                break
        while index2 == index1:
            rand  = random.uniform(0, 1)
            for j in range(0, len(input)):
                if fitness_input[j] > rand:
                    index2 = j
                    break
        tmp1 = input[index1].copy()
        tmp2 = input[index2].copy()

        # 然后选取两个下标x和y，交换[x+1, y]之间的部分
        x = random.randint(0, 62)
        y = random.randint(x + 1, 63)
        tmp1_c = list(tmp1[:x+1]) + list(tmp2[x+1:y+1]) + list(tmp1[y+1:])
        tmp2_c = list(tmp2[:x+1]) + list(tmp1[x+1:y+1]) + list(tmp2[y+1:])
        output.append(tmp1_c)
        output.append(tmp2_c)

    print("crossed-over")
    return output

# Mutation 1: Changing tone for an octave.
# 将某个音变化一个八度，体现在数字上就是+7或-7，这里需要注意不能选到0和15
def change_for_octave(input, fitness_input, n): 
    output = []
    for i in range(n):
        rand = random.uniform(0, 1)
        index = 0
        for j in range(0, len(input)):
            if fitness_input[j] > rand:
                index = j
                break
        
        rand_index = random.randint(0, 63)
        while input[index][rand_index] == 0 or input[index][rand_index] == 15:
            rand_index = random.randint(0, 63)
        
        tmp = input[index].copy() # 注意深拷贝

        if tmp[rand_index] > 7:
            tmp[rand_index] -= 7
        else:
            tmp[rand_index] += 7

        output.append(tmp)
        
    print("changed for octave")
    return output


# Mutation 2: Changing one tone.
# 随机改变数组中的某个数，这里不必考虑那个被改变的数是不是15，但是第一个数不能是15
def change_one_note(input, fitness_input, n): 
    output = []
    for i in range(n):
        rand = random.uniform(0, 1)
        index = 0
        for j in range(0, len(input)):
            if fitness_input[j] > rand:
                index = j
                break
        
        rand_index = random.randint(0, 63)
        #print(rand, index, rand_index, input[index])
        
        tmp = input[index].copy()
        tmp[rand_index] = random.randint(0, 15)
        while rand_index == 0 and tmp[rand_index] == 15:
            tmp[rand_index] = random.randint(0, 15)
        output.append(tmp)

    print("changed one note")
    return output


# Mutation 3: Swapping two consecutive notes.
# 交换两个音符，既然是音符，就要考虑不能抽到15
def swap_two_notes(input, fitness_input, n): 
    output = []
    for i in range(n):
        rand = random.uniform(0, 1)
        index = 0
        for j in range(1, len(input)):
            if fitness_input[j] > rand:
                index = j
                break
            
        notation = note2syn_time(input[index].copy())
        swap_index = random.randint(1,len(notation)-1)
        notation[swap_index], notation[swap_index-1] = notation[swap_index-1], notation[swap_index] #与前面的互换
        output.append(syn_time2note(notation))
    print("swapped two notes")
    return output


# 随机生成n个旋律（如果有种子的话就不需要这个函数了）
# 按文献里的表示方法，将所有旋律限制在某个特定大调（例如C大调）的两个八度之内
# 不考虑升降号
def create_answer(n): 
    result = []
    for i in range(n):
        seed = np.random.choice(range(0,16), 64)
        while seed[0] == 15:
            seed = np.random.choice(range(0,16), 64)
        result.append(seed)
    return result

def genetic(input,iteration):
    for t in range(iteration):
        print("Current: ", t)
        print("length: ", len(input))
        fitness_input = selection(input)

        output_crossover = crossover(input, fitness_input, 50)
        input = list(input) + output_crossover
        print("length: ", len(input))
        fitness_input = selection(input)
        
        output_octave = change_for_octave(input, fitness_input, 100)
        output_note = change_one_note(input, fitness_input, 100)
        output_swap = swap_two_notes(input, fitness_input, 100)
        
        output_trans = transposition(input, fitness_input, 100)
        output_invers = inversion(input, fitness_input, 100)
        output_retro = retrograde(input, fitness_input, 100)


        mutated_population = list(output_octave) + list(output_note) + list(output_swap) + list(output_trans) + list(output_invers) + list(output_retro)

        fitness_mutated = fitness_evaluation(mutated_population)

        to_sort = []
        for k in range(len(mutated_population)):
            to_sort.append([mutated_population[k], fitness_mutated[k]])
        mutated_sorted = sorted(to_sort, key = lambda x:x[1], reverse=True)
        
        input = []
        for k in range(20):
            input.append(mutated_sorted[k][0].copy())
        print(input[0])
        syn = note2syn(input[0])
        r_stop = False
        for note,time in syn: #pysynth不可以处理休止切分
            if note == 'r' and time < 0:
                r_stop = True
                print("r_stop!",syn)
        if not r_stop:
            ps.make_wav(syn,fn=f"middle/dong_star{t}.wav")
    return input[0]


##----------------------以董小姐及小星星作为种子生成结果并可视化适应度函数分量------------------------
reference_melody = [1,1,5,5,6,6,5,15,4,4,3,3,2,2,1,15,5,5,4,4,3,3,2,15,5,5,4,4,3,3,2,15,1,1,5,5,6,6,5,15,4,4,3,3,2,2,1,15,5,5,4,4,3,3,2,15,5,5,4,4,3,3,2,15]
ref_melody = np.array(reference_melody)

#董小姐和小星星旋律作为种子
dong = [0, 15, 15, 15, 5, 15, 8, 6, 15, 15, 15, 15, 0, 6, 6, 6, 7, 15, 5, 15, 5, 15, 6, 5, 15, 3, 15, 15, 0, 2, 3, 2, 3, 15, 2, 5, 5, 15, 15, 15, 0, 15, 15, 2, 5, 15, 15, 2, 5, 3, 15, 15, 5, 3, 15, 15, 5, 3, 15, 15, 5, 3, 15, 15]
# ps.make_wav(note2syn(dong),fn="dong.wav")
input = [dong.copy() for _ in range(5)] + [reference_melody.copy() for _ in range(5)]
result = genetic(input,100)
ps.make_wav(note2syn(result),fn="dong_test.wav")

#适应度函数分量可视化
plt.plot(average_log(whole_log),label=['repeat_score','inner_harm_score', 'refer_score'])
plt.legend()
plt.savefig("whole.png")
plt.plot(average_log(chord_log),label=['chord_ev','chord_match'])
plt.legend()
plt.savefig("chord.png")
plt.plot(average_log(mel_rhy_log),label=['mel','rhy'])
plt.legend()
plt.savefig("mel_rhy.png")
