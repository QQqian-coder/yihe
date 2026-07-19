// charts.js — 益和AI爆款引擎 落地价值量化图表
// 图1：AI赋能前后关键指标对比
(function () {
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim() || '#6366f1';
  var accent2 = style.getPropertyValue('--accent2').trim() || '#ec4899';
  var accent3 = style.getPropertyValue('--accent3').trim() || '#10b981';
  var ink = style.getPropertyValue('--ink').trim() || '#1a1a2e';
  var muted = style.getPropertyValue('--muted').trim() || '#6b7280';
  var rule = style.getPropertyValue('--rule').trim() || '#e5e7eb';
  var bg2 = style.getPropertyValue('--bg2').trim() || '#ffffff';

  // 等待 ECharts 与 DOM 就绪
  function init() {
    var dom = document.getElementById('chart-value');
    if (!dom || typeof echarts === 'undefined') {
      if (typeof echarts === 'undefined') {
        console.warn('[charts.js] ECharts 未加载，图表无法渲染');
      }
      return;
    }
    var chart = echarts.init(dom, null, { renderer: 'canvas' });

    // 数据：以"AI赋能前"为基准 100，"AI赋能后"为相对值
    // 同时保留原始数值用于标签展示
    var metrics = [
      {
        name: '配方设计周期',
        before: 100,
        after: 15,              // 2周 / 3.5月 ≈ 0.5/3.5 ≈ 14.3%
        beforeLabel: '3.5 个月',
        afterLabel: '2 周',
        reduction: '-85%'
      },
      {
        name: '功效验证活体使用量',
        before: 100,
        after: 40,               // 40% / 100%
        beforeLabel: '100%（全活体）',
        afterLabel: '40%（AI预筛60%）',
        reduction: '-60%'
      },
      {
        name: '单配方验证成本',
        before: 100,
        after: 25,               // 2万 / 8万 = 25%
        beforeLabel: '约 8 万元',
        afterLabel: '约 2 万元',
        reduction: '-75%'
      },
      {
        name: '新品上市周期',
        before: 100,
        after: 25,               // 1.5月 / 6月 = 25%
        beforeLabel: '6 个月',
        afterLabel: '1.5 个月',
        reduction: '-75%'
      }
    ];

    var option = {
      backgroundColor: bg2,
      grid: {
        left: 12,
        right: 28,
        top: 90,
        bottom: 60,
        containLabel: true
      },
      legend: {
        top: 24,
        right: 24,
        itemWidth: 14,
        itemHeight: 14,
        itemGap: 20,
        textStyle: {
          color: ink,
          fontSize: 13
        },
        data: ['AI赋能前（基准）', 'AI赋能后']
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        backgroundColor: 'rgba(26,26,46,0.92)',
        borderColor: 'transparent',
        textStyle: { color: '#fff', fontSize: 13 },
        padding: [10, 14],
        formatter: function (params) {
          var idx = params[0].dataIndex;
          var m = metrics[idx];
          var lines = [];
          lines.push('<div style="font-weight:700;margin-bottom:6px;">' + m.name + '</div>');
          params.forEach(function (p) {
            var valLabel = p.seriesName.indexOf('前') > -1 ? m.beforeLabel : m.afterLabel;
            lines.push(
              '<div style="display:flex;align-items:center;gap:8px;margin:3px 0;">' +
              '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:' + p.color + ';"></span>' +
              '<span>' + p.seriesName + '：</span>' +
              '<span style="font-weight:700;">' + valLabel + '</span>' +
              '</div>'
            );
          });
          lines.push(
            '<div style="margin-top:6px;padding-top:6px;border-top:1px solid rgba(255,255,255,0.2);color:#a5f3fc;font-weight:700;">' +
            '降幅 ' + m.reduction +
            '</div>'
          );
          return lines.join('');
        }
      },
      xAxis: {
        type: 'category',
        data: metrics.map(function (m) { return m.name; }),
        axisLine: { lineStyle: { color: rule } },
        axisTick: { show: false },
        axisLabel: {
          color: muted,
          fontSize: 12.5,
          interval: 0,
          lineHeight: 18,
          rich: {}
        }
      },
      yAxis: {
        type: 'value',
        name: '相对值（AI赋能前=100）',
        nameTextStyle: {
          color: muted,
          fontSize: 11.5,
          padding: [0, 0, 0, 30]
        },
        max: 110,
        axisLine: { show: false },
        axisTick: { show: false },
        splitLine: { lineStyle: { color: rule, type: 'dashed' } },
        axisLabel: { color: muted, fontSize: 11 }
      },
      series: [
        {
          name: 'AI赋能前（基准）',
          type: 'bar',
          barWidth: 22,
          barGap: '20%',
          itemStyle: {
            color: accent2,
            borderRadius: [4, 4, 0, 0]
          },
          label: {
            show: true,
            position: 'top',
            color: accent2,
            fontSize: 11,
            fontWeight: 700,
            formatter: function (p) {
              return metrics[p.dataIndex].beforeLabel;
            }
          },
          data: metrics.map(function (m) { return m.before; })
        },
        {
          name: 'AI赋能后',
          type: 'bar',
          barWidth: 22,
          itemStyle: {
            color: accent3,
            borderRadius: [4, 4, 0, 0]
          },
          label: {
            show: true,
            position: 'top',
            color: accent3,
            fontSize: 11,
            fontWeight: 700,
            formatter: function (p) {
              return metrics[p.dataIndex].afterLabel;
            }
          },
          data: metrics.map(function (m) { return m.after; }),
          // 降幅标注：用 markPoint 在柱顶上方再标一个降幅徽标
          markPoint: {
            symbol: 'pin',
            symbolSize: 0,
            label: {
              show: true,
              position: 'top',
              distance: -18,
              color: '#fff',
              backgroundColor: accent,
              padding: [4, 8],
              borderRadius: 6,
              fontSize: 11.5,
              fontWeight: 700,
              formatter: function (p) {
                return metrics[p.dataIndex].reduction;
              }
            },
            data: metrics.map(function (m, i) {
              return { coord: [i, m.after + 12] };
            })
          }
        }
      ]
    };

    chart.setOption(option);

    // 响应式
    window.addEventListener('resize', function () {
      chart.resize();
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
