import pandas as pd
from rdkit import Chem


def find_most_boron_molecule(csv_file):
    """
    找出BACE数据集中硼原子(B)最多的分子
    """
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    total_molecules = len(df)

    print(f"BACE数据集总共有 {total_molecules} 个分子")
    print("正在查找硼原子最多的分子...")
    print("=" * 60)

    # 初始化变量
    processed_count = 0
    max_boron_count = 0
    max_boron_molecules = []  # 存储所有具有最多硼原子的分子
    boron_distribution = {}  # 统计硼原子数量的分布

    # 遍历每一个分子
    for idx, row in df.iterrows():
        processed_count += 1

        # 显示进度
        if processed_count % 1000 == 0:
            print(f"已处理 {processed_count}/{total_molecules} 个分子...")

        smiles = row['smiles']
        mol_id = row['mol_id']

        # 跳过空值
        if pd.isna(smiles):
            continue

        try:
            # 从SMILES创建分子对象
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is not None:
                # 统计所有原子的数量
                atom_counts = {}
                for atom in mol.GetAtoms():
                    symbol = atom.GetSymbol()
                    atom_counts[symbol] = atom_counts.get(symbol, 0) + 1

                # 获取硼原子数量
                boron_count = atom_counts.get('B', 0)

                # 更新硼原子分布统计
                boron_distribution[boron_count] = boron_distribution.get(boron_count, 0) + 1

                # 检查是否是当前硼原子最多的分子
                if boron_count > max_boron_count:
                    max_boron_count = boron_count
                    max_boron_molecules = [{
                        'mol_id': mol_id,
                        'smiles': smiles,
                        'atom_counts': atom_counts,
                        'boron_count': boron_count
                    }]
                    print(f"发现新的最多硼原子记录: {boron_count} 个硼原子")
                    print(f"  Mol_ID: {mol_id}")
                    print(f"  SMILES: {smiles}")
                    print(f"  原子组成: {atom_counts}")
                    print("-" * 40)
                elif boron_count == max_boron_count and boron_count > 0:
                    # 如果硼原子数量与当前最大值相同，也记录下来
                    max_boron_molecules.append({
                        'mol_id': mol_id,
                        'smiles': smiles,
                        'atom_counts': atom_counts,
                        'boron_count': boron_count
                    })

        except Exception as e:
            # 处理过程中出现错误，跳过
            continue

    # 打印总结
    print("=" * 60)
    print("搜索完成!")
    print(f"总共处理了 {processed_count} 个分子")

    # 打印硼原子分布
    print("\n硼原子数量分布:")
    print("-" * 30)
    for boron_count in sorted(boron_distribution.keys()):
        if boron_count > 0:
            print(f"  B{boron_count}: {boron_distribution[boron_count]} 个分子")

    # 打印最多硼原子的分子
    if max_boron_molecules:
        print(f"\n找到 {len(max_boron_molecules)} 个含有最多硼原子的分子:")
        print(f"硼原子数量: {max_boron_count}")
        print("-" * 60)

        for i, mol in enumerate(max_boron_molecules, 1):
            print(f"分子 #{i}:")
            print(f"  Mol_ID: {mol['mol_id']}")
            print(f"  SMILES: {mol['smiles']}")
            print(f"  原子组成: {mol['atom_counts']}")
            print(f"  硼原子数量: {mol['boron_count']}")
            print("-" * 40)
    else:
        print("没有找到含有硼原子的分子")

    return max_boron_molecules


def analyze_boron_molecules(csv_file):
    """
    分析BACE数据集中含硼分子的详细信息
    """
    df = pd.read_csv(csv_file)

    print("分析含硼分子:")
    print("=" * 60)

    boron_molecules = []
    boron_atom_counts = []

    for idx, row in df.iterrows():
        smiles = row['smiles']

        try:
            mol = Chem.MolFromSmiles(str(smiles))
            if mol:
                atom_counts = {}
                for atom in mol.GetAtoms():
                    symbol = atom.GetSymbol()
                    atom_counts[symbol] = atom_counts.get(symbol, 0) + 1

                boron_count = atom_counts.get('B', 0)
                if boron_count > 0:
                    boron_molecules.append({
                        'mol_id': row['mol_id'],
                        'smiles': smiles,
                        'atom_counts': atom_counts,
                        'boron_count': boron_count
                    })
                    boron_atom_counts.append(boron_count)
        except:
            continue

    print(f"数据集中含硼分子的总数: {len(boron_molecules)}")

    if boron_molecules:
        # 统计硼原子数量的分布
        print(f"硼原子数量范围: {min(boron_atom_counts)} - {max(boron_atom_counts)}")
        print(f"平均硼原子数量: {sum(boron_atom_counts) / len(boron_atom_counts):.2f}")

        # 按硼原子数量排序
        boron_molecules.sort(key=lambda x: x['boron_count'], reverse=True)

        print(f"\n硼原子数量最多的前10个分子:")
        print("-" * 60)
        for i, mol in enumerate(boron_molecules[:10], 1):
            print(f"#{i}: B{mol['boron_count']}")
            print(f"  Mol_ID: {mol['mol_id']}")
            print(f"  SMILES: {mol['smiles']}")
            print(f"  原子组成: {mol['atom_counts']}")
            print("-" * 40)

    return boron_molecules


# 使用示例
if __name__ == "__main__":
    csv_file_path = 'data/bace.csv'  # 替换为您的文件路径

    # 首先分析所有含硼分子
    print("第一步: 分析所有含硼分子")
    boron_mols = analyze_boron_molecules(csv_file_path)

    print("\n" + "=" * 60 + "\n")

    # 然后找出硼原子最多的分子
    print("第二步: 找出硼原子最多的分子")
    most_boron_molecules = find_most_boron_molecule(csv_file_path)

    # 如果找到了分子，可以将结果保存到文件
    if most_boron_molecules:
        result_df = pd.DataFrame(most_boron_molecules)
        result_df.to_csv('most_boron_molecules_in_bace.csv', index=False)
        print(f"\n结果已保存到 most_boron_molecules_in_bace.csv")

        # 输出最重要的结果
        print("\n" + "=" * 60)
        print("最终结果:")
        print(f"BACE数据集中硼原子最多的分子有 {len(most_boron_molecules)} 个")
        print(f"每个分子含有 {most_boron_molecules[0]['boron_count']} 个硼原子")
        print(f"其中一个分子的SMILES: {most_boron_molecules[0]['smiles']}")