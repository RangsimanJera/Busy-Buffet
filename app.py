import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Busy Buffet Analysis")

# --- Data Loading & Global Preparation ---
file_name = 'busy_buffet_data.xlsx'

@st.cache_data
def load_and_prepare_data(file_name):
    all_sheets = pd.read_excel(file_name, sheet_name=None)
    df_list = []
    for sheet_name, df_s in all_sheets.items():
        df_s['date_sheet'] = str(sheet_name)
        df_list.append(df_s)

    df = pd.concat(df_list, ignore_index=True)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    time_cols = ['queue_start', 'queue_end', 'meal_start', 'meal_end']
    for col in time_cols:
        df[col] = pd.to_datetime(df[col], format='%H:%M:%S', errors='coerce')

    df['wait_time_mins'] = (df['queue_end'] - df['queue_start']).dt.total_seconds() / 60.0
    df['meal_time_mins'] = (df['meal_end'] - df['meal_start']).dt.total_seconds() / 60.0
    df['is_walkaway'] = df['queue_start'].notna() & df['meal_start'].isna()

    def assign_status(row):
        if pd.notna(row['queue_start']) and pd.isna(row['meal_start']):
            return 'Walk-away'
        elif pd.notna(row['queue_start']) and pd.notna(row['meal_start']):
            return 'Waited & Seated'
        elif pd.isna(row['queue_start']) and pd.notna(row['meal_start']):
            return 'Direct Seating'
        else:
            return 'No Usable Data'

    df['customer_status'] = df.apply(assign_status, axis=1)
    return df

df = load_and_prepare_data(file_name)

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "1. Data Preparation", 
    "2. Task 1: Evaluate Staff Comments", 
    "3. Task 2: Disprove Recommended Actions", 
    "4. Task 3: Support a Recommended Action"
])

# --- Page 1: Data Preparation ---
if page == "1. Data Preparation":
    st.title('Busy Buffet Analysis and Recommendations')
    st.header('1. Data Preparation')
    st.markdown('นำข้อมูลจากทุก Sheet มารวมกันแยกด้วยตัวแปรใหม่ตามชื่อ sheet และสร้างตัวแปรใหม่อื่น ๆ  สำหรับการวิเคราะห์ ได้แก่ `wait_time_mins`, `meal_time_mins` และสถานะ `is_walkaway`')
    st.subheader('Prepared Data Sample')
    st.dataframe(df.head())

# --- Page 2: Task 1 ---
elif page == "2. Task 1: Evaluate Staff Comments":
    st.header('2. Task 1: Evaluate Staff Comments')

    # Comment 1
    st.markdown('## Comment 1')
    st.markdown('**"In-house (hotel) customers are unhappy that they have to wait for a table. Walk-in customers are also unhappy, when they queue up for a long time and leave the queue because they don’t want to wait any longer".**')

    # 1. กราฟแจกแจงระยะเวลารอคิว
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    guest_order = ['Walk in', 'In house']
    df['Guest_type'] = pd.Categorical(df['Guest_type'], categories=guest_order, ordered=True)
    sns.boxplot(data=df[df['wait_time_mins'].notna()], x='Guest_type', y='wait_time_mins', palette="Set2", ax=ax1, order=guest_order)
    ax1.set_title('Distribution of Wait Time (In-house vs Walk-in)')
    ax1.set_ylabel('Wait Time (Minutes)')
    ax1.set_xlabel('Guest Type')
    grouped = df[df['wait_time_mins'].notna()].groupby('Guest_type', observed=False)
    for i, guest_type_val in enumerate(guest_order):
        if guest_type_val in grouped.groups:
            group = grouped.get_group(guest_type_val)
            median = group['wait_time_mins'].median()
            count = group['wait_time_mins'].count()
            ax1.text(i, median, f'median={median:.1f}\nn={count}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')
    st.pyplot(fig1)
    plt.close(fig1)

    # 2. กราฟ: Walk-away vs TOTAL Guests
    total_status = df.groupby(['Guest_type', 'is_walkaway']).size().unstack(fill_value=0)
    total_status.rename(columns={False: 'Got Table (Direct + Waited)', True: 'Walk-away'}, inplace=True)
    total_percent = total_status.div(total_status.sum(axis=1), axis=0) * 100
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    total_percent.plot(kind='bar', stacked=True, color=['#2ca02c', '#d62728'], ax=ax2)
    ax2.set_title('Walk-away Proportion vs TOTAL Guests')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_xlabel('Guest Type')
    plt.xticks(rotation=0)
    for c in ax2.containers:
        labels = [f'{v.get_height():.1f}%' if v.get_height() > 0 else '' for v in c]
        ax2.bar_label(c, labels=labels, label_type='center', color='white', weight='bold')
    st.pyplot(fig2)
    plt.close(fig2)

    # 3. กราฟ: Walk-away vs QUEUED Guests
    df_queued = df[df['queue_start'].notna()]
    queued_status = df_queued.groupby(['Guest_type', 'customer_status']).size().unstack(fill_value=0)
    queued_status['Waited & Seated'] = queued_status.get('Waited & Seated', 0)
    queued_status['Walk-away'] = queued_status.get('Walk-away', 0)
    queued_status = queued_status[['Waited & Seated', 'Walk-away']]
    queued_percent = queued_status.div(queued_status.sum(axis=1), axis=0) * 100
    combined_queued_status = queued_status.sum(axis=0).to_frame().T
    combined_queued_status.index = ['Combined']
    combined_queued_percent = combined_queued_status.div(combined_queued_status.sum(axis=1), axis=0) * 100
    queued_percent_for_plot = pd.concat([queued_percent, combined_queued_percent])
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    queued_percent_for_plot.plot(kind='bar', stacked=True, color=['#ff7f0e', '#d62728'], ax=ax3)
    ax3.set_title('Walk-away Proportion vs QUEUED Guests Only (Including Combined)')
    ax3.set_ylabel('Percentage (%)')
    ax3.set_xlabel('Guest Type')
    plt.xticks(rotation=0)
    for c in ax3.containers:
        labels = [f'{v.get_height():.2f}%' if v.get_height() > 0 else '' for v in c]
        ax3.bar_label(c, labels=labels, label_type='center', color='white', weight='bold')
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    st.markdown("**Conclusion:**\nความคิดเห็นนี้ **เป็นความจริง** ข้อมูลยืนยันว่าลูกค้า In-house ต้องรอคิวจริง เฉลี่ยเกือบ 30 นาที ซึ่งนานเกินไปสำหรับแขกโรงแรม ในขณะเดียวกันฝั่งลูกค้า Walk-in ก็มีเวลาเฉลี่ยการรอคิวสูงถึงราว 45 นาที นอกจากนี้สัดส่วนการทิ้งคิวโดยรวมของทั้งลูกค้า In-house และ Wlak-in อยู่ที่ 19.18% จึงยืนยันได้ว่าลูกค้ารอไม่ไหวและเกิดความไม่พึงพอใจตามที่พนักงานแจ้ง")

    # Comment 2
    st.markdown('## Comment 2')
    st.markdown('**"We are very busy every day of the week. If it’s going to be this busy every week I think it’s impossible to sustain this business. This buffet business is not possible for this hotel".**')
    daily_status = df.groupby(['date_sheet', 'customer_status']).size().unstack(fill_value=0)
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    daily_status[['Direct Seating', 'Waited & Seated', 'Walk-away']].plot(kind='bar', stacked=True, color=['#2ca02c', '#ff7f0e', '#d62728'], ax=ax4)
    ax4.set_title('Daily Customer Flow (Are we really busy everyday?)')
    ax4.set_ylabel('Number of Groups')
    ax4.set_xlabel('Date (Sheet Name)')
    plt.xticks(rotation=0)
    st.pyplot(fig4)
    plt.close(fig4)
    st.markdown("**Conclusion:**\nควาามคิดเห็นพนักงานข้อนี้ **ไม่เป็นความจริง** จากกราฟแสดงให้เห็นชัดเจนว่าร้านไม่ได้ยุ่งทุกวัน ในวันที่ 133, 173 และ 183 ลูกค้าทุกคนสามารถได้โต๊ะทันทีโดยไม่ต้องมีการรอคิวเลย มีเพียงวันที่ 143 และ 153 ที่ลูกค้าต้องมีการรอคิวและมีลูกค้าที่ลุกออกไปเท่านั้น")

    # Comment 3
    st.markdown('## Comment 3')
    st.markdown('**"Walk-in customers sit the whole day. It\'s very difficult to find seats for in-house customers. We don\'t have enough tables so when one customer sits for a long time it makes the queue very long".**')
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x='meal_time_mins', hue='Guest_type', element="step", stat="density", common_norm=False, bins=20, ax=ax5)
    ax5.axvline(120, color='red', linestyle='--', label='2 Hours Mark')
    ax5.set_title('Distribution of Meal Duration (Does Walk-in sit all day?)')
    ax5.set_xlabel('Meal Time (Minutes)')
    ax5.legend()
    st.pyplot(fig5)
    plt.close(fig5)
    st.markdown("**Conclusion:**\nความคิดเห็นนี้เป็นจริงบาวส่วน จากกราฟแม้ว่าลูกค้า Walk-in จะใช้เวลานั่งทานนานกว่า In-house จริง (เฉลี่ย 73 นาที เทียบกับ 45 นาที) แต่ความคิดเห็นที่ว่า \"นั่งแช่ทั้งวัน\" นั้น **กล่าวเกินจริง** เพราะกว่า 89.45% ของลูกค้า Walk-in ทานเสร็จภายในเวลาไม่เกิน 2 ชั่วโมง เมื่อวิเคราะห์ร่วมกับกราฟ \"Daily Customer Flow\" ในข้อ Comment 2 ปัญหาคิวที่ยาวไม่ได้มาจากการที่ลูกค้านั่งแช่ แต่มาจากปริมาณคนที่เข้ามาพร้อมกันเกินจำนวนโต๊ะมากกว่า")

# --- Page 3: Task 2 ---
elif page == "3. Task 2: Disprove Recommended Actions":
    st.header('3. Task 2: Disprove Recommended Actions')
    st.markdown('For each of the recommended actions, create visual and analysis to **disprove why each of them will not work**')

    # Recommended 1: Reduce seating time
    st.markdown('## Recommended 1')
    st.markdown('**Reduce seating time (5 hours to less)**')
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    sns.histplot(df['meal_time_mins'].dropna(), bins=30, kde=True, ax=ax6)
    ax6.set_title('Distribution of All Customer Meal Durations')
    ax6.set_xlabel('Meal Duration (Minutes)')
    ax6.set_ylabel('Number of Customers')
    ax6.grid(axis='y', alpha=0.75)
    st.pyplot(fig6)
    plt.close(fig6)
    st.markdown("**Conclusion:**\nจากกราฟจะเห็นได้ว่าลูกค้าส่วนใหญ่ใช้เวลานั่งทานอยู่ที่ประมาณ 40 - 90 นาที ดังนั้นข้อเสนอที่ให้ลดเวลานั่งทานลงจาก 5 ชั่ว จึงแทบไม่มีผลต่อระบบการจัดการลูกค้า อีกทั้งยังลดทอนความน่าดึงดูดของการตลาดอีกด้วย")

    # Recommended 2: Increase price
    st.markdown('## Recommened 2')
    st.markdown('**Increase price everyday to 259**')
    def assign_day_type(sheet):
        if str(sheet) in ['143', '153']:
            return 'Holiday (Weekend)'
        return 'Weekday'
    df['day_type'] = df['date_sheet'].apply(assign_day_type)
    customer_volume_by_day_type = df.groupby('day_type')['pax'].sum().reset_index()
    customer_volume_by_day_type = customer_volume_by_day_type.sort_values(by='pax', ascending=False)
    fig7, ax7 = plt.subplots(figsize=(8, 5))
    sns.barplot(x='day_type', y='pax', data=customer_volume_by_day_type, palette='pastel', ax=ax7, hue='day_type', legend=False)
    ax7.set_title('Total Customer Volume: Holidays vs. Weekdays')
    ax7.set_xlabel('Day Type')
    ax7.set_ylabel('Total Number of Pax')
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig7)
    plt.close(fig7)
    st.markdown("**Conclusion:**\nจากข้อมูลถ้าสมมติว่า 143 และ 153 เป็นวันหยุดจากสภาพที่มี่การเข้าใช้บริการของลูกค้าเป็นจำนวนมาก และลองเทียบกับปฏิทินเดือนมีนาคมที่ผ่านมา (ปี 2569) ที่วันที่ 14 และ 15 เป็นวันเสาร์และอาทิตย์ตามลำดับ แล้วให้ข้อมูลในวันอื่น ๆ เป็นวันธรรมดา\n\nจะพบว่าจากกราฟจำนวนลูกค้าที่พบว่า Weekday และ Holiday มีจำนวนลูกค้าใกล้เคียงกัน แสดงว่าร้านมีฐานลูกค้าวันธรรมดาที่ค่อนข้างแข็งแรง ดังนั้นการปรับราคาบุฟเฟต์ขึ้นเป็น 259 บาททุกวันจากเดิม 159 บาทในวันธรรมดาและ 199 บาทในวันหยุด อาจส่งผลเสียหลักต่อกลุ่มลูกค้าวันธรรมดา ทำให้จำนวนลูกค้าลดลง ความถี่ในการกลับมาใช้บริการลดลง")

    # Recommended 3: Queue skipping
    st.markdown('## Recommended 3')
    st.markdown('**Queue skipping for in-house guest**')
    daily_pax_by_guest_type = df.groupby(['date_sheet', 'Guest_type'])['pax'].sum().unstack(fill_value=0)
    fig8, ax8 = plt.subplots(figsize=(10, 6))
    daily_pax_by_guest_type.plot(kind='bar', stacked=False, color=['#ff7f0e', '#1f77b4'], ax=ax8)
    ax8.set_title('Daily Customer Volume (Pax): Walk-in vs In-house')
    ax8.set_xlabel('Date')
    ax8.set_ylabel('Total Pax')
    ax8.tick_params(axis='x', rotation=0)
    ax8.legend(title='Guest Type')
    ax8.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    st.pyplot(fig8)
    plt.close(fig8)
    st.markdown("**Conclusion:**\nจากกราฟจะเห็นว่าลูกค้า Walk-in มีจำนวนมากกว่า In-house โดยเฉพาะในวันที่มีคิวจำนวนมาก (143 และ 153) ดังนั้นหากร้านใช้ระบบ “skipping queue” โดยให้ลูกค้า In-house ได้สิทธิ์ข้ามคิวก่อน อาจส่งผลเสียต่อประสบการณ์ของลูกค้า Walk-in ที่เป็นลูกค้าหลักอย่างชัดเจน โดยเฉพาะในช่วงที่ลูกค้า Walk-in มีจำนวนสูง เพราะจะทำให้เวลารอคิวยาวขึ้น ความรู้สึกไม่ยุติธรรมเพิ่มขึ้น และมีโอกาสเกิด walk-away มากขึ้นกว่าเดิม")

# --- Page 4: Task 3 ---
elif page == "4. Task 3: Support a Recommended Action":
    st.header('4. Task 3: Support a Recommended Action')
    st.markdown('Pick 1 recommended action, create visual and analysis to support why it will work. Describe and add through your own experience why you believe it to be the best solution. You can adjust specific details of the action and explain your reasoning.')

    st.subheader('Model 1: Walk-away Prediction (Feature Engineering & Importance)')
    # Feature Engineering สำหรับโมเดล
    df['is_walkin'] = (df['Guest_type'] == 'Walk in').astype(int)
    df['is_busy_day'] = df['date_sheet'].isin(['143', '153']).astype(int)
    df['arrival_time'] = df['queue_start'].fillna(df['meal_start'])
    df['arrival_hour'] = df['arrival_time'].dt.hour
    df['pax'] = df['pax'].fillna(df['pax'].median())
    features = ['pax', 'is_walkin', 'is_busy_day', 'arrival_hour']

    st.markdown("### Walk-away Prediction Model")
    df_queue = df[df['queue_start'].notna()].copy()
    df_queue['is_walkaway'] = df_queue['is_walkaway'].astype(int)
    X_clf = df_queue[features]
    y_clf = df_queue['is_walkaway']
    clf_model = RandomForestClassifier(random_state=42, max_depth=4)
    clf_model.fit(X_clf, y_clf)
    clf_importance = pd.DataFrame({'Feature': features, 'Importance': clf_model.feature_importances_})
    st.write("**Feature Importance (Walk-away - Full Data):**")
    st.dataframe(clf_importance.sort_values(by='Importance', ascending=False))

    st.markdown("### Walk-aways by Pax Group Size")
    walkaway_pax_counts = df[df['is_walkaway'] == True]['pax'].value_counts().reset_index()
    walkaway_pax_counts.columns = ['Pax', 'Walkaway_Count']
    st.write("จำนวนลูกค้า (Pax) ที่เดินหนีบ่อยที่สุด:")
    st.dataframe(walkaway_pax_counts)
    fig9, ax9 = plt.subplots(figsize=(8, 5))
    sns.barplot(x='Pax', y='Walkaway_Count', data=walkaway_pax_counts, palette='viridis', hue='Pax', legend=False)
    ax9.set_title('Number of Walk-aways by Pax Group Size')
    ax9.set_xlabel('Number of Pax')
    ax9.set_ylabel('Number of Walk-aways')
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig9)
    plt.close(fig9)

    st.markdown("### Wait Time Prediction Model")
    df_wait = df[(df['queue_start'].notna()) & (df['wait_time_mins'].notna())].copy()
    X_reg = df_wait[features]
    y_reg = df_wait['wait_time_mins']
    reg_model = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=4)
    reg_model.fit(X_reg, y_reg)
    reg_importance = pd.DataFrame({'Feature': features, 'Importance': reg_model.feature_importances_})
    st.write("**Feature Importance (Wait Time - Full Data):**")
    st.dataframe(reg_importance.sort_values(by='Importance', ascending=False))

    st.markdown("### Average Wait Time by Arrival Hour")
    average_wait_time_by_hour = df.groupby('arrival_hour')['wait_time_mins'].mean().reset_index()
    st.write("เวลารอคิวเฉลี่ย (นาที) ตามชั่วโมงที่มาถึง:")
    st.dataframe(average_wait_time_by_hour.sort_values(by='wait_time_mins', ascending=False))
    fig10, ax10 = plt.subplots(figsize=(10, 6))
    sns.barplot(x='arrival_hour', y='wait_time_mins', data=average_wait_time_by_hour.sort_values(by='wait_time_mins', ascending=False), palette='magma', hue='arrival_hour', legend=False)
    ax10.set_title('Average Wait Time by Arrival Hour')
    ax10.set_xlabel('Arrival Hour')
    ax10.set_ylabel('Average Wait Time (Minutes)')
    plt.xticks(rotation=45)
    ax10.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig10)
    plt.close(fig10)

    # Timeline Visualization
    st.subheader('Visualizing Table Utilization vs Queue Growth (Busiest Day - Sheet 153)')
    busy_day_task3_1 = df[df['date_sheet'] == '153'].copy()
    time_cols_task3_1 = ['queue_start', 'queue_end', 'meal_start', 'meal_end']
    for col in time_cols_task3_1:
        busy_day_task3_1[col] = pd.to_datetime('2023-01-01 ' + busy_day_task3_1[col].astype(str).str.split(' ').str[-1], errors='coerce')

    timeline_task3_1 = pd.date_range(start='2023-01-01 06:00:00', end='2023-01-01 12:00:00', freq='1min')
    active_tables_task3_1 = []
    active_queues_task3_1 = []

    for t in timeline_task3_1:
        seated_task3_1 = busy_day_task3_1[(busy_day_task3_1['meal_start'] <= t) & (busy_day_task3_1['meal_end'] > t)].shape[0]
        active_tables_task3_1.append(seated_task3_1)
        
        queued_task3_1 = busy_day_task3_1[
            (busy_day_task3_1['queue_start'] <= t) & 
            (
                ((busy_day_task3_1['queue_end'] > t) & busy_day_task3_1['queue_end'].notna()) |
                ((busy_day_task3_1['meal_start'] > t) & busy_day_task3_1['queue_end'].isna() & busy_day_task3_1['meal_start'].notna())
            )
        ].shape[0]
        active_queues_task3_1.append(queued_task3_1)

    df_timeline_task3_1 = pd.DataFrame({'Time': timeline_task3_1, 'Occupied Tables': active_tables_task3_1, 'Groups in Queue': active_queues_task3_1})
    max_capacity_task3_1 = df_timeline_task3_1['Occupied Tables'].max()

    fig_task3_1, ax_task3_1 = plt.subplots(figsize=(12, 6))
    ax_task3_1.plot(df_timeline_task3_1['Time'], df_timeline_task3_1['Occupied Tables'], label='Occupied Tables', color='#2ca02c', linewidth=3)
    ax_task3_1.plot(df_timeline_task3_1['Time'], df_timeline_task3_1['Groups in Queue'], label='Groups in Queue', color='#d62728', linewidth=3)
    ax_task3_1.fill_between(df_timeline_task3_1['Time'], df_timeline_task3_1['Groups in Queue'], color='#d62728', alpha=0.1)
    ax_task3_1.axhline(max_capacity_task3_1, color='grey', linestyle='--', label=f'Max Capacity Hit ({max_capacity_task3_1} groups)')
    ax_task3_1.axvspan(pd.to_datetime('2023-01-01 08:15:00'), pd.to_datetime('2023-01-01 09:30:00'), color='yellow', alpha=0.2, label='Peak Queue (Bottleneck)')
    ax_task3_1.set_title('Table Utilization vs Queue Growth (Busiest Day - Sheet 153)', fontsize=16, fontweight='bold')
    ax_task3_1.set_xlabel('Time of Day', fontsize=12)
    ax_task3_1.set_ylabel('Number of Customer Groups', fontsize=12)
    ax_task3_1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=0)
    ax_task3_1.legend(loc='upper left', fontsize=11)
    ax_task3_1.grid(True, alpha=0.3)
    st.pyplot(fig_task3_1)
    plt.close(fig_task3_1)

    # Detailed Zone Timeline
    st.subheader('Detailed Available Tables by Zone vs Customer Queue over Time (Sheet 153)')
    busy_day_task3_2 = df[df['date_sheet'] == '153'].copy()
    time_cols_task3_2 = ['queue_start', 'queue_end', 'meal_start', 'meal_end']
    for col in time_cols_task3_2:
        busy_day_task3_2[col] = pd.to_datetime('2023-01-01 ' + busy_day_task3_2[col].astype(str).str.split(' ').str[-1], errors='coerce')

    def assign_zone_and_units_task3_2(table_str):
        if pd.isna(table_str) or str(table_str) == '99': return pd.Series([None, 0])
        parts = str(table_str).split('-')
        units_used = len(parts)
        num_match = re.match(r'(\d+)', parts[0])
        if num_match:
            table_num = int(num_match.group(1))
            if 1 <= table_num <= 6: zone = 'Indoor'
            elif 7 <= table_num <= 15: zone = 'Outdoor'
            else: zone = None
        else: zone = None
        return pd.Series([zone, units_used])

    busy_day_task3_2[['Zone', 'Units_Used']] = busy_day_task3_2['table_no.'].apply(assign_zone_and_units_task3_2)
    timeline_task3_2 = pd.date_range(start='2023-01-01 06:00:00', end='2023-01-01 12:00:00', freq='1min')
    MAX_INDOOR, MAX_OUTDOOR = 12, 17
    indoor_avail_task3_2, outdoor_avail_task3_2, inhouse_queue_task3_2, walkin_queue_task3_2 = [], [], [], []

    for t in timeline_task3_2:
        seated_task3_2 = busy_day_task3_2[(busy_day_task3_2['meal_start'] <= t) & (busy_day_task3_2['meal_end'] > t)]
        indoor_used_task3_2 = seated_task3_2[seated_task3_2['Zone'] == 'Indoor']['Units_Used'].sum()
        outdoor_used_task3_2 = seated_task3_2[seated_task3_2['Zone'] == 'Outdoor']['Units_Used'].sum()
        indoor_avail_task3_2.append(MAX_INDOOR - indoor_used_task3_2)
        outdoor_avail_task3_2.append(MAX_OUTDOOR - outdoor_used_task3_2)
        
        queued_task3_2 = busy_day_task3_2[
            (busy_day_task3_2['queue_start'] <= t) &
            (
                ((busy_day_task3_2['queue_end'] > t) & busy_day_task3_2['queue_end'].notna()) |
                ((busy_day_task3_2['meal_start'] > t) & busy_day_task3_2['queue_end'].isna() & busy_day_task3_2['meal_start'].notna())
            )
        ]
        inhouse_queue_task3_2.append(queued_task3_2[queued_task3_2['Guest_type'] == 'In house'].shape[0])
        walkin_queue_task3_2.append(queued_task3_2[queued_task3_2['Guest_type'] == 'Walk in'].shape[0])

    df_timeline_task3_2 = pd.DataFrame({'Time': timeline_task3_2, 'Indoor Available': indoor_avail_task3_2, 'Outdoor Available': outdoor_avail_task3_2, 'In-house Queue': inhouse_queue_task3_2, 'Walk-in Queue': walkin_queue_task3_2})

    fig_solution_task3_2, (ax1_solution_task3_2, ax2_solution_task3_2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    ax1_solution_task3_2.plot(df_timeline_task3_2['Time'], df_timeline_task3_2['Indoor Available'], label='Indoor Available (Max 12)', color='#1f77b4', linewidth=2)
    ax1_solution_task3_2.plot(df_timeline_task3_2['Time'], df_timeline_task3_2['Outdoor Available'], label='Outdoor Available (Max 17)', color='#ff7f0e', linewidth=2)
    ax1_solution_task3_2.set_title('Available Tables by Zone vs Customer Queue over Time (Sheet 153)', fontsize=16, fontweight='bold')
    ax1_solution_task3_2.set_ylabel('Number of Available Tables', fontsize=12)
    ax1_solution_task3_2.legend(loc='upper right')
    ax1_solution_task3_2.grid(True, alpha=0.3)
    ax1_solution_task3_2.fill_between(df_timeline_task3_2['Time'], 0, df_timeline_task3_2['Indoor Available'], alpha=0.1, color='#1f77b4')
    ax1_solution_task3_2.fill_between(df_timeline_task3_2['Time'], 0, df_timeline_task3_2['Outdoor Available'], alpha=0.1, color='#ff7f0e')

    ax2_solution_task3_2.plot(df_timeline_task3_2['Time'], df_timeline_task3_2['In-house Queue'], label='In-house Queue', color='#2ca02c', linewidth=2)
    ax2_solution_task3_2.plot(df_timeline_task3_2['Time'], df_timeline_task3_2['Walk-in Queue'], label='Walk-in Queue', color='#d62728', linewidth=2)
    ax2_solution_task3_2.set_ylabel('Groups in Queue', fontsize=12)
    ax2_solution_task3_2.set_xlabel('Time of Day', fontsize=12)
    ax2_solution_task3_2.legend(loc='upper right')
    ax2_solution_task3_2.grid(True, alpha=0.3)
    ax2_solution_task3_2.fill_between(df_timeline_task3_2['Time'], 0, df_timeline_task3_2['In-house Queue'], alpha=0.2, color='#2ca02c')
    ax2_solution_task3_2.fill_between(df_timeline_task3_2['Time'], 0, df_timeline_task3_2['Walk-in Queue'], alpha=0.2, color='#d62728')

    peak_start_task3_2 = pd.to_datetime('2023-01-01 08:00:00')
    peak_end_task3_2 = pd.to_datetime('2023-01-01 09:30:00')
    ax1_solution_task3_2.axvspan(peak_start_task3_2, peak_end_task3_2, color='yellow', alpha=0.15)
    ax2_solution_task3_2.axvspan(peak_start_task3_2, peak_end_task3_2, color='yellow', alpha=0.15, label='Peak Bottleneck')
    ax2_solution_task3_2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.tight_layout()
    st.pyplot(fig_solution_task3_2)
    plt.close(fig_solution_task3_2)

    st.subheader('Proposed Solution: Optimized Zone Allocation')
    st.markdown("**Conclusion:**\nจากการวิเคราะห์ทั้งหมด ผมขอเลือกปรับปรุงข้อเสนอ \"Queue Skipping for In-house Guest\" ให้เป็นรูปแบบการจัดสรร Zone การรับประทานอาหารแทน โดย:\n\n1. ตั้งเริ่มต้นให้พื้นที่โซน Indoor (12 โต๊ะ) เป็น Priority Zone สำหรับลูกค้า In-house ในช่วง Peak Hour\n2. จัดสรรพื้นที่โซน Outdoor (17 โต๊ะ) สำหรับลูกค้า Walk-in\n3. หากโซน In-door ว่างในช่วงที่ไม่มีลูกค้า In-house ก็สามารถดึงคิว Walk-in เข้าไปนั่งได้")
    st.markdown('**เหตุผล:**')
    st.markdown('1. **Solving "In-house Unhappiness" Directly:** กราฟชี้ให้เห็นว่าในช่วง Peak (08:00-09:30 น.) ความต้องการของแขก In-house (เส้นสีเขียว) อยู่ในระดับที่โซน Indoor (12 โต๊ะ) สามารถรองรับได้เกือบทั้งหมด การจัดสรรพื้นที่ให้ชัดเจนจะทำให้แขกโรงแรมแทบไม่ต้องรอคิว หรือรอในระยะเวลาที่สั้นมาก ซึ่งตอบโจทย์ Pain Point ข้อแรกของพนักงาน.')
    st.markdown('2. **Preventing "Queue Starvation":** หากเราเลือกใช้วิธีลัดคิวแบบปกติ (Reommended 3 เดิม) คิวของ Walk-in จะถูกแช่แข็ง จนทำให้ Walk-away rate พุ่งสูงขึ้น แต่การแยกโซน Outdoor (17 โต๊ะ) ไว้ให้ Walk-in จะช่วยให้คิวสีแดงยังคงขยับตัวได้เรื่อย ๆ แม้จะเป็นช่วงที่ In-house หนาแน่นก็ตาม')
    st.markdown('3. **Optimizing Operational Flow:** การแบ่งโซนคือการจัดสรรลูกค้าอย่างมีประสิทธิภาพ โซน Indoor ที่มีเครื่องปรับอากาศจะช่วยรักษาภาพลักษณ์ของโรงแรมต่อลูกค้า In-house ในขณะที่โซน Outdoor จะช่วยบริหารจัดการลูกค้าปริมาณมากที่มาจาก TikTok ได้อย่างมีประสิทธิภาพ')