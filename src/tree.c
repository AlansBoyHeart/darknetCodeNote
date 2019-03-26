#include <stdio.h>
#include <stdlib.h>
#include "tree.h"
#include "utils.h"
#include "data.h"

void change_leaves(tree *t, char *leaf_list)
{
    list *llist = get_paths(leaf_list);
    char **leaves = (char **)list_to_array(llist);
    int n = llist->size;
    int i,j;
    int found = 0;
    for(i = 0; i < t->n; ++i){
        t->leaf[i] = 0;
        for(j = 0; j < n; ++j){
            if (0==strcmp(t->name[i], leaves[j])){
                t->leaf[i] = 1;
                ++found;
                break;
            }
        }
    }
    fprintf(stderr, "Found %d leaves.\n", found);
}

float get_hierarchy_probability(float *x, tree *hier, int c, int stride)
{
    float p = 1;
    while(c >= 0){
        p = p * x[c*stride];
        c = hier->parent[c];
    }
    return p;
}

//参数为（预测结果的指针+类别的索引（偏移量），9418，l.softmax_tree的地址，0,17*17）
void hierarchy_predictions(float *predictions, int n, tree *hier, int only_leaves, int stride)
{
    int j;
    for(j = 0; j < n; ++j){
        int parent = hier->parent[j];
        if(parent >= 0){
            predictions[j*stride] *= predictions[parent*stride];  //每个节点的最终概率值是自己节点的概率值乘以所有父节点的概率值的结果。
        }
    }
    if(only_leaves){
        for(j = 0; j < n; ++j){
            if(!hier->leaf[j]) predictions[j*stride] = 0;
        }
    }
}

//参数为（预测结果的指针+类别的索引（偏移量），l.softmax_tree的地址,tree_thresh树的阈值，17*17）
int hierarchy_top_prediction(float *predictions, tree *hier, float thresh, int stride)
{
    float p = 1;
    int group = 0;
    int i;
    while(1){
        float max = 0;
        int max_i = 0;

        for(i = 0; i < hier->group_size[group]; ++i){
            int index = i + hier->group_offset[group];
            float val = predictions[(i + hier->group_offset[group])*stride];
            if(val > max){
                max_i = index;
                max = val;
            }
        }
        if(p*max > thresh){
            p = p*max;
            group = hier->child[max_i];
            if(hier->child[max_i] < 0) return max_i;  //如果当前节点的孩子节点的个数小于0（是-1）则返回类别索引
        } else if (group == 0){
            return max_i;
        } else {
            return hier->parent[hier->group_offset[group]];
        }
    }
    return 0;
}

tree *read_tree(char *filename)
{
    tree t = {0};  //初始化结构体tree，第一个参数赋值为0，其他的参数执行默认的初始化参数。
    FILE *fp = fopen(filename, "r");

    char *line;
    int last_parent = -1;
    int group_size = 0;
    int groups = 0;
    int n = 0;
    while((line=fgetl(fp)) != 0){  //fgetl()函数的功能是读取文件中的一行
        char *id = calloc(256, sizeof(char));
        int parent = -1;
        sscanf(line, "%s %d", id, &parent);  //sscanf从指定的line中提取指定格式的字符，赋值到id，&parent
        t.parent = realloc(t.parent, (n+1)*sizeof(int));
        t.parent[n] = parent;  //t.parent[4]=0  t.parent[5]=0

        t.child = realloc(t.child, (n+1)*sizeof(int));
        t.child[n] = -1;

        t.name = realloc(t.name, (n+1)*sizeof(char *));
        t.name[n] = id;
        if(parent != last_parent){  //0!=-1  0!=0
            ++groups;  // group=1
            t.group_offset = realloc(t.group_offset, groups * sizeof(int));
            t.group_offset[groups - 1] = n - group_size;  //group的偏移量[0] =0 (4-4),当前读取到文件的第五行，下标为4的行.offset抵消，补偿，形成分支。
            t.group_size = realloc(t.group_size, groups * sizeof(int));
            t.group_size[groups - 1] = group_size;  //group_size[0] = 4
            group_size = 0;  //group_size=0
            last_parent = parent;  //当前点的父节点，last_parent = 0
        }
        t.group = realloc(t.group, (n+1)*sizeof(int));
        t.group[n] = groups;  //t.group[4] = 1 ,t.group[5]==1
        if (parent >= 0) {
            t.child[parent] = groups;  //t.child[0]=1  t.child[0]=1
        }
        ++n;  //n=5  n=6
        ++group_size;  //group_size=1 group_size=2
    }
    ++groups;
    t.group_offset = realloc(t.group_offset, groups * sizeof(int));
    t.group_offset[groups - 1] = n - group_size;
    t.group_size = realloc(t.group_size, groups * sizeof(int));
    t.group_size[groups - 1] = group_size;
    t.n = n;
    t.groups = groups;
    t.leaf = calloc(n, sizeof(int));
    int i;
    for(i = 0; i < n; ++i) t.leaf[i] = 1;
    for(i = 0; i < n; ++i) if(t.parent[i] >= 0) t.leaf[t.parent[i]] = 0;

    fclose(fp);
    tree *tree_ptr = calloc(1, sizeof(tree));
    *tree_ptr = t;
    //error(0);
    return tree_ptr;
}
