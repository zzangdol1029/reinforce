// 발표자료 생성 (pptxgenjs) — 포털 WAS 자원 동시 튜닝 RL
const pptxgen = require("pptxgenjs");
const p = new pptxgen();
p.defineLayout({ name: "W", width: 13.333, height: 7.5 });
p.layout = "W";
p.author = "김완수";
p.title = "강화학습 기반 포털 WAS 자원 동시 튜닝";

// ── 팔레트 ─────────────────────────────────────────────
const INK = "12233F";     // 네이비(다크 슬라이드/제목)
const INK2 = "1E3A5F";
const TEAL = "0E7490";     // 주 강조
const CYAN = "06B6D4";
const MINT = "10B981";     // 좋음/오라클
const RED = "E11D48";      // 현재 설정 문제
const AMBER = "F59E0B";
const TEXT = "1F2937";
const MUTE = "6B7280";
const CARD = "F1F5F9";
const ICE = "DBEAFE";
const WHITE = "FFFFFF";
const F = "Malgun Gothic"; // 한글 폰트

const W = 13.333, H = 7.5, ML = 0.7;
const shadow = () => ({ type: "outer", color: "000000", blur: 7, offset: 3, angle: 135, opacity: 0.16 });

function contentHeader(s, num, kicker, title) {
  s.background = { color: WHITE };
  // 섹션 번호 원
  s.addShape(p.shapes.OVAL, { x: ML, y: 0.55, w: 0.62, h: 0.62, fill: { color: TEAL } });
  s.addText(String(num), { x: ML, y: 0.55, w: 0.62, h: 0.62, align: "center", valign: "middle", color: WHITE, bold: true, fontFace: F, fontSize: 24 });
  s.addText(kicker, { x: ML + 0.8, y: 0.5, w: 11, h: 0.3, color: TEAL, bold: true, fontFace: F, fontSize: 12, charSpacing: 2, margin: 0 });
  s.addText(title, { x: ML + 0.8, y: 0.76, w: 11.6, h: 0.6, color: INK, bold: true, fontFace: F, fontSize: 26, margin: 0 });
}

// ════════════════════════════════════════════════════════
// S1 — 타이틀
// ════════════════════════════════════════════════════════
let s = p.addSlide();
s.background = { color: INK };
s.addShape(p.shapes.RECTANGLE, { x: 0, y: 0, w: 0.28, h: H, fill: { color: TEAL } });
s.addShape(p.shapes.OVAL, { x: 10.2, y: -1.6, w: 4.6, h: 4.6, fill: { color: INK2 } });
s.addShape(p.shapes.OVAL, { x: 11.4, y: 3.6, w: 3.2, h: 3.2, fill: { color: TEAL, transparency: 55 } });
s.addText("강화학습 적용 프로젝트", { x: ML + 0.2, y: 1.5, w: 10, h: 0.4, color: CYAN, bold: true, fontFace: F, fontSize: 15, charSpacing: 3 });
s.addText("강화학습 기반 포털 WAS\n자원(워커 + DB풀) 동시 튜닝", { x: ML + 0.2, y: 1.95, w: 11.2, h: 1.8, color: WHITE, bold: true, fontFace: F, fontSize: 40, lineSpacingMultiple: 1.05 });
s.addText("Undertow 워커 스레드풀 + HikariCP DB 커넥션풀의 부하 적응형 동적 최적화", { x: ML + 0.2, y: 3.9, w: 11, h: 0.5, color: ICE, fontFace: F, fontSize: 17 });
s.addText([
  { text: "DQN · PPO · SAC 비교", options: { color: WHITE, bold: true } },
  { text: "   |   실제 3-포털 사양 기반 Gym 시뮬레이터", options: { color: ICE } },
], { x: ML + 0.2, y: 4.45, w: 11, h: 0.4, fontFace: F, fontSize: 14 });
s.addShape(p.shapes.LINE, { x: ML + 0.2, y: 5.55, w: 4.2, h: 0, line: { color: TEAL, width: 2 } });
s.addText("김완수  ·  충북대학교 산업인공지능학과", { x: ML + 0.2, y: 5.7, w: 11, h: 0.4, color: WHITE, fontFace: F, fontSize: 14 });

// ════════════════════════════════════════════════════════
// S2 — 목차
// ════════════════════════════════════════════════════════
s = p.addSlide();
s.background = { color: WHITE };
s.addText("목차", { x: ML, y: 0.55, w: 6, h: 0.7, color: INK, bold: true, fontFace: F, fontSize: 30 });
s.addText("CONTENTS", { x: ML, y: 1.2, w: 6, h: 0.3, color: TEAL, bold: true, fontFace: F, fontSize: 12, charSpacing: 3 });
const toc = [
  ["01", "적용 문제", "MSA 포털의 자원 설정 한계 · 왜 RL인가"],
  ["02", "모델 정의", "State · Action · Reward · 시뮬레이터"],
  ["03", "방법 / 알고리즘", "DQN · PPO · SAC"],
  ["04", "프로그램 소스 설명", "환경 · 베이스라인 · 학습 · 평가"],
  ["05", "결과 제시 / 분석", "현재 설정 vs 자원 최적 · RL 성능"],
  ["06", "기여도 / 향후 연구", "기여점 · 한계 · 확장 방향"],
];
let ty = 1.9;
toc.forEach((t, i) => {
  const yy = ty + i * 0.83;
  s.addText(t[0], { x: ML, y: yy, w: 0.9, h: 0.7, color: TEAL, bold: true, fontFace: F, fontSize: 30, margin: 0, valign: "middle" });
  s.addText(t[1], { x: ML + 1.05, y: yy, w: 4.2, h: 0.4, color: INK, bold: true, fontFace: F, fontSize: 18, margin: 0, valign: "middle" });
  s.addText(t[2], { x: ML + 1.05, y: yy + 0.36, w: 7.5, h: 0.32, color: MUTE, fontFace: F, fontSize: 12.5, margin: 0 });
});
s.addShape(p.shapes.RECTANGLE, { x: 9.7, y: 1.9, w: 0.04, h: 5.0, fill: { color: ICE } });
s.addText("적용 문제부터\n향후 연구까지\n6단계로 구성", { x: 10.0, y: 3.4, w: 2.9, h: 1.6, color: INK2, bold: true, fontFace: F, fontSize: 18, align: "left" });

// ════════════════════════════════════════════════════════
// S3 — 적용 문제 (구조 + 현재설정 문제)
// ════════════════════════════════════════════════════════
s = p.addSlide();
contentHeader(s, 1, "적용 문제 · PROBLEM", "MSA 포털 시스템과 현재 자원 설정의 한계");
// 좌: 아키텍처 흐름
s.addText("시스템 구조 (고정 · 단일 서버 · 비공유)", { x: ML, y: 1.7, w: 6, h: 0.35, color: TEAL, bold: true, fontFace: F, fontSize: 14 });
const flow = [["nginx", "WEB"], ["Gateway / Eureka", "포털중계"], ["Undertow WAS", "3개 포털"]];
flow.forEach((f, i) => {
  const yy = 2.2 + i * 1.15;
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: ML, y: yy, w: 4.6, h: 0.9, fill: { color: i === 2 ? TEAL : CARD }, line: { color: i === 2 ? TEAL : "CBD5E1", width: 1 }, rectRadius: 0.08, shadow: shadow() });
  s.addText([{ text: f[0] + "  ", options: { bold: true, color: i === 2 ? WHITE : INK, fontSize: 16 } }, { text: f[1], options: { color: i === 2 ? ICE : MUTE, fontSize: 12 } }], { x: ML + 0.25, y: yy, w: 4.1, h: 0.9, fontFace: F, valign: "middle", margin: 0 });
  if (i < 2) s.addText("▼", { x: ML + 2.1, y: yy + 0.88, w: 0.4, h: 0.27, color: MUTE, fontSize: 12, align: "center" });
});
s.addText("사용자 ×2  ·  연구 ×4  ·  관리 ×2  (N중화 고정)", { x: ML, y: 5.75, w: 4.8, h: 0.4, color: INK2, italic: true, fontFace: F, fontSize: 12.5 });
// 우: 문제 카드
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 6.1, y: 1.95, w: 6.55, h: 4.5, fill: { color: "FEF2F3" }, line: { color: RED, width: 1.5 }, rectRadius: 0.1, shadow: shadow() });
s.addText("현재: 자원을 '정적'으로 고정", { x: 6.45, y: 2.2, w: 6, h: 0.45, color: RED, bold: true, fontFace: F, fontSize: 18 });
s.addText([
  { text: "워커 스레드·DB 커넥션풀을 한 번 박아두고 운영", options: { bullet: true, breakLine: true } },
  { text: "예) 연구포털: 워커 2048 / DB 150 정적 고정", options: { bullet: true, breakLine: true, color: RED, bold: true } },
  { text: "부하·요청 믹스는 시간대별로 크게 변동", options: { bullet: true, breakLine: true } },
  { text: "→ 한가할 땐 과다(메모리 낭비), 몰릴 땐 병목", options: { bullet: true, breakLine: true } },
  { text: "스케일링 불가(단일 서버) → '자원 노브'를 최적화해야", options: { bullet: true } },
], { x: 6.55, y: 2.75, w: 6.0, h: 3.5, color: TEXT, fontFace: F, fontSize: 14.5, paraSpaceAfter: 10, lineSpacingMultiple: 1.02 });

// ════════════════════════════════════════════════════════
// S4 — 왜 RL: 결합 병목
// ════════════════════════════════════════════════════════
s = p.addSlide();
contentHeader(s, 1, "적용 문제 · WHY RL", "두 자원의 '결합 병목' — 단순 규칙이 약한 지점");
// 병목 다이어그램: 3박스 → min
const bx = [["CPU 한계", "코어 수 / CPU시간", INK2], ["워커 한계", "W / 체류시간", TEAL], ["DB 한계", "D / DB시간", CYAN]];
bx.forEach((b, i) => {
  const xx = ML + i * 2.35;
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: xx, y: 1.95, w: 2.1, h: 1.15, fill: { color: b[2] }, rectRadius: 0.08, shadow: shadow() });
  s.addText([{ text: b[0] + "\n", options: { bold: true, fontSize: 15, color: WHITE } }, { text: b[1], options: { fontSize: 11, color: ICE } }], { x: xx, y: 1.95, w: 2.1, h: 1.15, align: "center", valign: "middle", fontFace: F, margin: 0 });
});
s.addText("min", { x: ML + 7.0, y: 1.95, w: 1.0, h: 1.15, align: "center", valign: "middle", color: RED, bold: true, fontFace: F, fontSize: 26 });
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: ML + 8.05, y: 1.95, w: 3.3, h: 1.15, fill: { color: "ECFDF5" }, line: { color: MINT, width: 1.5 }, rectRadius: 0.08 });
s.addText([{ text: "처리능력\n", options: { bold: true, fontSize: 14, color: MINT } }, { text: "= 셋 중 최소 = 병목", options: { fontSize: 11, color: TEXT } }], { x: ML + 8.05, y: 1.95, w: 3.3, h: 1.15, align: "center", valign: "middle", fontFace: F, margin: 0 });
// 핵심 포인트
const pts = [
  ["워커만 늘려도 소용없다", "DB풀이 작으면 워커가 DB 커넥션 대기로 묶여 낭비"],
  ["DB풀을 무작정 키울 수 없다", "공유 통합DB(max 1500)가 과부하 → 오히려 느려짐"],
  ["요청 믹스는 관측 불가", "체류시간·DB비중을 점유율·지연으로 '추론'해야 함"],
  ["선제 조절(예측)이 필요", "부하가 오른 뒤 반응하면 늦음 → 장기·다목적·부분관측"],
];
pts.forEach((t, i) => {
  const xx = ML + (i % 2) * 6.1, yy = 3.6 + Math.floor(i / 2) * 1.55;
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: xx, y: yy, w: 5.8, h: 1.35, fill: { color: CARD }, rectRadius: 0.08 });
  s.addShape(p.shapes.RECTANGLE, { x: xx, y: yy, w: 0.08, h: 1.35, fill: { color: TEAL } });
  s.addText(t[0], { x: xx + 0.25, y: yy + 0.16, w: 5.4, h: 0.4, color: INK, bold: true, fontFace: F, fontSize: 15, margin: 0 });
  s.addText(t[1], { x: xx + 0.25, y: yy + 0.62, w: 5.4, h: 0.6, color: MUTE, fontFace: F, fontSize: 12.5, margin: 0 });
});

// ════════════════════════════════════════════════════════
// S5 — 모델: State / Action
// ════════════════════════════════════════════════════════
s = p.addSlide();
contentHeader(s, 2, "모델 · MDP", "State · Action 정의");
s.addText("State (10차원 — 모두 관측 가능한 메트릭)", { x: ML, y: 1.7, w: 8, h: 0.35, color: TEAL, bold: true, fontFace: F, fontSize: 15 });
const stateRows = [
  ["부하", "정규화 도착률, 도착률 추세"],
  ["자원 점유", "CPU 이용률, 워커풀 점유율, DB풀 점유율"],
  ["성능", "정규화 지연, 지연 추세, 직전 드롭율"],
  ["현재 설정", "W / maxW, D / maxD"],
];
s.addTable(stateRows.map(r => [
  { text: r[0], options: { fill: { color: INK }, color: WHITE, bold: true, fontSize: 13, valign: "middle", align: "center" } },
  { text: r[1], options: { fill: { color: CARD }, color: TEXT, fontSize: 13, valign: "middle" } },
]), { x: ML, y: 2.1, w: 7.0, colW: [1.7, 5.3], rowH: 0.52, fontFace: F, border: { type: "solid", color: WHITE, pt: 2 } });
s.addText("※ 요청 믹스(체류시간·DB비중)는 비노출 → 추론 (부분관측)", { x: ML, y: 4.4, w: 7.0, h: 0.4, color: MUTE, italic: true, fontFace: F, fontSize: 11.5 });

s.addText("Action — 두 자원 동시 조절", { x: 8.05, y: 1.7, w: 4.6, h: 0.35, color: TEAL, bold: true, fontFace: F, fontSize: 15 });
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 8.05, y: 2.1, w: 4.6, h: 1.05, fill: { color: CARD }, rectRadius: 0.08 });
s.addText([{ text: "이산 (DQN · PPO)\n", options: { bold: true, color: INK, fontSize: 14 } }, { text: "Discrete(25) = ΔW 5단계 × ΔD 5단계", options: { color: TEXT, fontSize: 12.5 } }], { x: 8.3, y: 2.1, w: 4.1, h: 1.05, valign: "middle", fontFace: F, margin: 0 });
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 8.05, y: 3.3, w: 4.6, h: 1.05, fill: { color: CARD }, rectRadius: 0.08 });
s.addText([{ text: "연속 (SAC)\n", options: { bold: true, color: INK, fontSize: 14 } }, { text: "Box(−1,1,(2,)) → [ΔW, ΔD]", options: { color: TEXT, fontSize: 12.5 } }], { x: 8.3, y: 3.3, w: 4.1, h: 1.05, valign: "middle", fontFace: F, margin: 0 });
s.addText("W = Undertow 워커 스레드 수\nD = HikariCP DB 커넥션 수", { x: 8.05, y: 4.5, w: 4.6, h: 0.7, color: INK2, fontFace: F, fontSize: 12.5 });

// Reward 바
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: ML, y: 5.55, w: 11.95, h: 1.2, fill: { color: INK }, rectRadius: 0.08, shadow: shadow() });
s.addText("Reward", { x: ML + 0.3, y: 5.7, w: 2, h: 0.4, color: CYAN, bold: true, fontFace: F, fontSize: 14 });
s.addText("− SLA위반(지연 초과) − 요청 드롭 − 워커 메모리 비용 − DB 커넥션 비용 − 잦은 변경(thrash)", { x: ML + 0.3, y: 6.08, w: 11.4, h: 0.6, color: WHITE, fontFace: F, fontSize: 15, valign: "top" });

// ════════════════════════════════════════════════════════
// S6 — 시뮬레이터
// ════════════════════════════════════════════════════════
s = p.addSlide();
contentHeader(s, 2, "모델 · SIMULATOR", "학습 환경 — 시뮬레이터 구성");
s.addText("RL은 실서버가 아니라 시뮬레이터에서 수십만 번 시행착오로 학습한다 (안전·고속)", { x: ML, y: 1.65, w: 12, h: 0.4, color: INK2, italic: true, fontFace: F, fontSize: 13.5 });
const simCards = [
  ["워크로드 생성", "도착률·요청믹스(체류시간·DB비중)를 시드 기반 결정적 trace로 생성 (일간 주기 + 변동)"],
  ["성능 모델 perf()", "처리능력 = min(CPU, 워커, DB) 병목 + 큐잉 지연 + 공유 DB 과부하 + 워커 과다 오버헤드"],
  ["보상 산출", "지연·드롭·메모리·DB비용을 한 숫자로 → 정책 학습 신호"],
  ["관측 메트릭", "CPU·워커풀·DB풀 점유율, 지연 → 에이전트가 병목 자원 진단"],
];
simCards.forEach((c, i) => {
  const yy = 2.25 + i * 1.12;
  s.addShape(p.shapes.OVAL, { x: ML, y: yy + 0.1, w: 0.5, h: 0.5, fill: { color: TEAL } });
  s.addText(String(i + 1), { x: ML, y: yy + 0.1, w: 0.5, h: 0.5, align: "center", valign: "middle", color: WHITE, bold: true, fontFace: F, fontSize: 16 });
  s.addText(c[0], { x: ML + 0.75, y: yy, w: 3.4, h: 0.7, color: INK, bold: true, fontFace: F, fontSize: 16, valign: "middle", margin: 0 });
  s.addShape(p.shapes.RECTANGLE, { x: ML + 4.2, y: yy + 0.05, w: 0.03, h: 0.6, fill: { color: ICE } });
  s.addText(c[1], { x: ML + 4.45, y: yy, w: 8.1, h: 0.75, color: TEXT, fontFace: F, fontSize: 13, valign: "middle", margin: 0 });
});

// ════════════════════════════════════════════════════════
// S7 — 방법/알고리즘
// ════════════════════════════════════════════════════════
s = p.addSlide();
contentHeader(s, 3, "방법 · ALGORITHMS", "DQN · PPO · SAC — 세 가지 적용 후 비교");
const algos = [
  ["DQN", "가치 기반 (Value)", "Q값을 학습해 최적 행동 선택. 이산 행동에 적합. 빠르지만 변동 큼.", INK2],
  ["PPO", "Actor-Critic (정책)", "정책을 안정적으로 갱신. 이산 행동. 안정적·범용.", TEAL],
  ["SAC", "Actor-Critic (연속)", "엔트로피 정규화로 탐색·안정성↑. 연속 행동(ΔW,ΔD)에 적합.", CYAN],
];
algos.forEach((a, i) => {
  const xx = ML + i * 4.05;
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: xx, y: 2.0, w: 3.75, h: 3.3, fill: { color: WHITE }, line: { color: "E2E8F0", width: 1 }, rectRadius: 0.1, shadow: shadow() });
  s.addShape(p.shapes.RECTANGLE, { x: xx, y: 2.0, w: 3.75, h: 0.95, fill: { color: a[3] } });
  s.addText(a[0], { x: xx, y: 2.0, w: 3.75, h: 0.95, align: "center", valign: "middle", color: WHITE, bold: true, fontFace: F, fontSize: 30, margin: 0 });
  s.addText(a[1], { x: xx + 0.25, y: 3.15, w: 3.25, h: 0.4, color: a[3], bold: true, fontFace: F, fontSize: 14.5, align: "center" });
  s.addText(a[2], { x: xx + 0.3, y: 3.65, w: 3.15, h: 1.5, color: TEXT, fontFace: F, fontSize: 13, align: "center", valign: "top", lineSpacingMultiple: 1.05 });
});
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: ML, y: 5.6, w: 11.95, h: 1.05, fill: { color: CARD }, rectRadius: 0.08 });
s.addText([{ text: "비교 관점:  ", options: { bold: true, color: INK } }, { text: "동일 시나리오에서 학습 → 평균 보상·SLA·자원사용량으로 비교하고 가장 좋은 알고리즘을 채택. 수렴 속도·안정성 차이도 분석.", options: { color: TEXT } }], { x: ML + 0.3, y: 5.6, w: 11.4, h: 1.05, valign: "middle", fontFace: F, fontSize: 13.5, margin: 0 });

// ════════════════════════════════════════════════════════
// S8 — 프로그램 소스
// ════════════════════════════════════════════════════════
s = p.addSlide();
contentHeader(s, 4, "소스 · IMPLEMENTATION", "프로그램 소스 구성");
const files = [
  ["envs/portal_env.py", "Gym 환경 — 3중 병목 perf(), 부분관측 상태, 보상"],
  ["config.py", "3개 포털 실사양 프리셋(연구/사용자/관리)"],
  ["baselines.py", "현재설정 · Static · HillClimb(반응형) · Oracle(상한)"],
  ["train.py", "DQN/PPO/SAC 학습 (Stable-Baselines3) + 학습곡선"],
  ["evaluate.py", "비교 표·그래프 + W·D vs 부하 시계열"],
];
files.forEach((fl, i) => {
  const yy = 1.95 + i * 0.62;
  s.addShape(p.shapes.RECTANGLE, { x: ML, y: yy, w: 4.6, h: 0.5, fill: { color: INK } });
  s.addText(fl[0], { x: ML + 0.15, y: yy, w: 4.4, h: 0.5, color: CYAN, fontFace: "Consolas", fontSize: 12.5, valign: "middle", margin: 0 });
  s.addText(fl[1], { x: ML + 4.8, y: yy, w: 7.7, h: 0.5, color: TEXT, fontFace: F, fontSize: 12.5, valign: "middle", margin: 0 });
});
// perf 핵심 코드 박스
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: ML, y: 5.15, w: 11.95, h: 1.55, fill: { color: "0F172A" }, rectRadius: 0.08, shadow: shadow() });
s.addText("핵심: 결합 병목 모델 (perf)", { x: ML + 0.3, y: 5.25, w: 8, h: 0.3, color: MINT, bold: true, fontFace: F, fontSize: 12.5 });
s.addText("capacity = min(cores/cpu_time,  W/mean_s_eff,  D/db_time_eff)\nlatency  = mean_s_eff / (1 − load/capacity) + 오버헤드", { x: ML + 0.3, y: 5.6, w: 11.4, h: 1.0, color: "E2E8F0", fontFace: "Consolas", fontSize: 13.5, valign: "top", lineSpacingMultiple: 1.1, margin: 0 });

// ════════════════════════════════════════════════════════
// S9 — 실제 사양
// ════════════════════════════════════════════════════════
s = p.addSlide();
contentHeader(s, 4, "소스 · REAL SPEC", "실제 3-포털 사양 반영 (자원설정 데이터)");
const hdr = ["포털", "노드", "노드당 사양", "Heap", "현재 워커", "현재 DB풀", "동접"];
const rows = [
  ["연구", "4대", "256c / 512GB", "256GB", "2048", "150", "1000"],
  ["사용자", "2대", "16c / 64GB", "31GB", "256(기본)", "100", "5000"],
  ["관리", "2대", "32c / 128GB", "31GB", "512(기본)", "50", "200"],
];
const tdata = [hdr.map(h => ({ text: h, options: { fill: { color: INK }, color: WHITE, bold: true, fontSize: 13, align: "center", valign: "middle" } }))];
rows.forEach((r, ri) => tdata.push(r.map((c, ci) => ({
  text: c,
  options: { fill: { color: ri % 2 ? CARD : WHITE }, color: (ci === 4 || ci === 5) ? RED : TEXT, bold: (ci === 0 || ci === 4 || ci === 5), fontSize: 13, align: ci === 0 ? "left" : "center", valign: "middle" }
}))));
s.addTable(tdata, { x: ML, y: 2.05, w: 11.95, colW: [1.5, 1.2, 2.6, 1.5, 1.9, 1.6, 1.65], rowH: 0.62, fontFace: F, border: { type: "solid", color: "E2E8F0", pt: 1 } });
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: ML, y: 4.75, w: 11.95, h: 1.85, fill: { color: "FEF2F3" }, line: { color: RED, width: 1.2 }, rectRadius: 0.08 });
s.addText("관찰 포인트", { x: ML + 0.3, y: 4.9, w: 6, h: 0.35, color: RED, bold: true, fontFace: F, fontSize: 14 });
s.addText([
  { text: "연구포털은 워커 2048을 정적 고정 → 대부분 과다(메모리 낭비), DB-heavy 구간엔 DB 150 부족", options: { bullet: true, breakLine: true } },
  { text: "사용자포털은 16코어로 동접 5000 → 피크에서 CPU 병목(풀 튜닝만으론 한계)", options: { bullet: true, breakLine: true } },
  { text: "공유 통합DB(Postgres, max 1500) → 포털별 DB풀 합이 과하면 공유 DB 과부하", options: { bullet: true } },
], { x: ML + 0.35, y: 5.3, w: 11.3, h: 1.25, color: TEXT, fontFace: F, fontSize: 12.5, paraSpaceAfter: 5 });

// ════════════════════════════════════════════════════════
// S10 — 결과 1: 현재설정 vs 자원 최적
// ════════════════════════════════════════════════════════
s = p.addSlide();
contentHeader(s, 5, "결과 · ANALYSIS", "현재 설정 vs 자원 최적(Oracle 상한)");
const rhdr = ["포털", "현재 (W/D)", "최적 (W/D)", "보상 개선", "효과"];
const rrows = [
  ["연구", "2048 / 150", "328 / 143", "+190", "워커 84%↓ 메모리 절감"],
  ["사용자", "256 / 100", "40 / 20", "+715", "과다 자원 대폭 정리"],
  ["관리", "512 / 50", "37 / 9", "+259", "거의 SLA 충족"],
];
const rt = [rhdr.map(h => ({ text: h, options: { fill: { color: TEAL }, color: WHITE, bold: true, fontSize: 13, align: "center", valign: "middle" } }))];
rrows.forEach((r, ri) => rt.push(r.map((c, ci) => ({
  text: c, options: { fill: { color: ri % 2 ? CARD : WHITE }, color: ci === 1 ? RED : (ci === 2 || ci === 3) ? MINT : TEXT, bold: (ci !== 4), fontSize: 13, align: ci === 0 ? "left" : "center", valign: "middle" }
}))));
s.addTable(rt, { x: ML, y: 2.0, w: 6.7, colW: [1.1, 1.6, 1.5, 1.2, 1.3], rowH: 0.66, fontFace: F, border: { type: "solid", color: "E2E8F0", pt: 1 } });
s.addText("현재 정적 설정은 모든 포털에서 비최적 — 워커 과다(메모리) + DB-heavy 구간 DB 부족", { x: ML, y: 4.85, w: 6.7, h: 0.7, color: INK2, italic: true, fontFace: F, fontSize: 12.5 });
// 차트: 워커 수 비교
s.addText("워커 스레드 수: 현재 vs 자원 최적", { x: 7.7, y: 1.9, w: 5, h: 0.35, color: INK, bold: true, fontFace: F, fontSize: 13 });
s.addChart(p.charts.BAR, [
  { name: "현재 설정", labels: ["연구", "사용자", "관리"], values: [2048, 256, 512] },
  { name: "자원 최적", labels: ["연구", "사용자", "관리"], values: [328, 40, 37] },
], {
  x: 7.6, y: 2.3, w: 5.2, h: 4.2, barDir: "col", chartColors: [RED, MINT],
  showValue: true, dataLabelColor: "334155", dataLabelFontSize: 9, dataLabelPosition: "outEnd",
  catAxisLabelColor: "64748B", valAxisLabelColor: "64748B", catAxisLabelFontSize: 11,
  valGridLine: { color: "EEF2F6", size: 0.5 }, showLegend: true, legendPos: "b", legendFontSize: 11,
  chartArea: { fill: { color: WHITE } },
});

// ════════════════════════════════════════════════════════
// S11 — 결과 2: RL 학습 (placeholder)
// ════════════════════════════════════════════════════════
s = p.addSlide();
contentHeader(s, 5, "결과 · RL TRAINING", "강화학습 성능 (학습 실행 후 삽입)");
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: ML, y: 1.95, w: 6.0, h: 3.0, fill: { color: CARD }, line: { color: "CBD5E1", width: 1, dashType: "dash" }, rectRadius: 0.08 });
s.addText("학습 곡선 (Reward vs Step)", { x: ML, y: 2.95, w: 6.0, h: 0.4, align: "center", color: MUTE, bold: true, fontFace: F, fontSize: 14 });
s.addText("results/learning_curve_<portal>_<algo>.png\n— train.py 실행 후 자동 생성", { x: ML, y: 3.4, w: 6.0, h: 0.8, align: "center", color: MUTE, fontFace: F, fontSize: 11.5 });
s.addText("예상 결과 / 분석 포인트", { x: 7.0, y: 1.95, w: 5.6, h: 0.4, color: TEAL, bold: true, fontFace: F, fontSize: 15 });
s.addText([
  { text: "RL이 현재설정·HillClimb(반응형)을 이기고 Oracle(상한)에 근접", options: { bullet: true, breakLine: true } },
  { text: "DQN vs PPO vs SAC 수렴 속도·안정성·최종 성능 비교 → 최적 채택", options: { bullet: true, breakLine: true } },
  { text: "W·D vs 부하 시계열: 부하 따라 자원을 선제 조절 (현재 설정 점선 대비)", options: { bullet: true, breakLine: true } },
  { text: "지표: 평균 보상 · SLA 위반율 · 평균 워커/DB · 메모리 절감률", options: { bullet: true } },
], { x: 7.0, y: 2.45, w: 5.6, h: 2.6, color: TEXT, fontFace: F, fontSize: 13.5, paraSpaceAfter: 9, lineSpacingMultiple: 1.03 });
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: ML, y: 5.3, w: 11.95, h: 1.35, fill: { color: INK }, rectRadius: 0.08, shadow: shadow() });
s.addText("실행 방법", { x: ML + 0.3, y: 5.45, w: 4, h: 0.3, color: CYAN, bold: true, fontFace: F, fontSize: 12.5 });
s.addText("python train.py --portal research --algo all --timesteps 250000   →   python evaluate.py --portal research", { x: ML + 0.3, y: 5.82, w: 11.4, h: 0.7, color: "E2E8F0", fontFace: "Consolas", fontSize: 13, valign: "top", margin: 0 });

// ════════════════════════════════════════════════════════
// S12 — 운영 적용 닫힌 루프 (continual)
// ════════════════════════════════════════════════════════
s = p.addSlide();
contentHeader(s, 6, "결론 · OPERATION LOOP", "운영 적용 — 닫힌 루프로 현실에 수렴");
const loop = [
  ["시뮬레이터 학습", "다양한 (W,D)·상황을 가상으로 탐색 → 정책 학습", TEAL],
  ["정책 배포", "실시간 지표 읽어 (W,D) 권고/적용", CYAN],
  ["운영 지표 수집", "점유율·지연 (Actuator·Micrometer·Hikari)", INK2],
  ["시뮬레이터 보정·재학습", "실측으로 sim-to-real 갭 축소", MINT],
];
const bw = 2.7, bh = 1.75, gy = 2.45, gap = (11.95 - bw * 4) / 3;
loop.forEach((l, i) => {
  const xx = ML + i * (bw + gap);
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: xx, y: gy, w: bw, h: bh, fill: { color: l[2] }, rectRadius: 0.1, shadow: shadow() });
  s.addShape(p.shapes.OVAL, { x: xx + 0.2, y: gy + 0.2, w: 0.5, h: 0.5, fill: { color: WHITE } });
  s.addText(String(i + 1), { x: xx + 0.2, y: gy + 0.2, w: 0.5, h: 0.5, align: "center", valign: "middle", color: l[2], bold: true, fontFace: F, fontSize: 18, margin: 0 });
  s.addText(l[0], { x: xx + 0.15, y: gy + 0.8, w: bw - 0.3, h: 0.45, align: "center", color: WHITE, bold: true, fontFace: F, fontSize: 14.5, margin: 0 });
  s.addText(l[1], { x: xx + 0.22, y: gy + 1.22, w: bw - 0.44, h: 0.5, align: "center", color: ICE, fontFace: F, fontSize: 10.5, margin: 0, valign: "top" });
  if (i < 3) s.addShape(p.shapes.LINE, { x: xx + bw + 0.05, y: gy + bh / 2, w: gap - 0.1, h: 0, line: { color: "94A3B8", width: 2.5, endArrowType: "triangle" } });
});
// 반환 루프 (4 → 1)
const cy = gy + bh + 0.5;
const x4 = ML + 3 * (bw + gap) + bw / 2, x1 = ML + bw / 2;
s.addShape(p.shapes.LINE, { x: x4, y: gy + bh, w: 0, h: cy - (gy + bh), line: { color: TEAL, width: 2.5 } });
s.addShape(p.shapes.LINE, { x: x1, y: cy, w: x4 - x1, h: 0, line: { color: TEAL, width: 2.5 } });
s.addShape(p.shapes.LINE, { x: x1, y: gy + bh, w: 0, h: cy - (gy + bh), line: { color: TEAL, width: 2.5, beginArrowType: "triangle" } });
s.addText("운영 데이터로 반복 개선 (continual learning)", { x: x1, y: cy + 0.06, w: x4 - x1, h: 0.35, align: "center", color: TEAL, bold: true, fontFace: F, fontSize: 12.5, margin: 0 });
// 핵심 메시지 바
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: ML, y: 5.75, w: 11.95, h: 1.0, fill: { color: INK }, rectRadius: 0.08, shadow: shadow() });
s.addText([
  { text: "핵심:  ", options: { bold: true, color: CYAN } },
  { text: "처음엔 가상(근사)에서 출발하지만 운영 데이터로 계속 보정 → 점차 현실 최적에 수렴 (디지털 트윈 + continual RL)", options: { color: WHITE } },
], { x: ML + 0.3, y: 5.75, w: 11.4, h: 1.0, valign: "middle", fontFace: F, fontSize: 14, margin: 0 });

// ════════════════════════════════════════════════════════
// S13 — 기여도 / 향후 / 한계
// ════════════════════════════════════════════════════════
s = p.addSlide();
contentHeader(s, 6, "결론 · CONTRIBUTION", "기여도 · 한계 · 향후 연구");
const cols = [
  ["기여도", MINT, [
    "실제 포털 사양 기반 2자원(워커+DB) 학습형 동시 튜닝",
    "현재 정적 설정 대비 메모리 대폭 절감 + SLA 유지",
    "관측 메트릭만으로 병목을 진단·선제 조절하는 정책",
  ]],
  ["한계", AMBER, [
    "사용자포털은 16코어 CPU 병목 → 코어/노드 증설 필요(별도 인사이트)",
    "도착률·요청믹스는 추정값 — 실측(APM/부하테스트) 보정 필요",
    "단일 노드 모델 — 공유 DB의 멀티포털 상호작용은 향후",
  ]],
  ["향후 연구", TEAL, [
    "실측 메트릭(Micrometer/JMX)으로 시뮬레이터 보정",
    "런타임 반영 또는 운영자 승인형 어드바이저 모드",
    "공유 DB 멀티포털 동시 최적화 · 온라인 학습",
  ]],
];
cols.forEach((c, i) => {
  const xx = ML + i * 4.05;
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: xx, y: 2.0, w: 3.75, h: 4.55, fill: { color: WHITE }, line: { color: "E2E8F0", width: 1 }, rectRadius: 0.1, shadow: shadow() });
  s.addShape(p.shapes.RECTANGLE, { x: xx, y: 2.0, w: 3.75, h: 0.7, fill: { color: c[1] } });
  s.addText(c[0], { x: xx, y: 2.0, w: 3.75, h: 0.7, align: "center", valign: "middle", color: WHITE, bold: true, fontFace: F, fontSize: 18, margin: 0 });
  s.addText(c[2].map((t, j) => ({ text: t, options: { bullet: true, breakLine: true, paraSpaceAfter: 12 } })), { x: xx + 0.3, y: 2.9, w: 3.2, h: 3.5, color: TEXT, fontFace: F, fontSize: 13, valign: "top", lineSpacingMultiple: 1.03 });
});

// ════════════════════════════════════════════════════════
// S13 — 감사합니다
// ════════════════════════════════════════════════════════
s = p.addSlide();
s.background = { color: INK };
s.addShape(p.shapes.RECTANGLE, { x: 0, y: 0, w: 0.28, h: H, fill: { color: TEAL } });
s.addShape(p.shapes.OVAL, { x: 10.4, y: 4.2, w: 4.4, h: 4.4, fill: { color: INK2 } });
s.addText("감사합니다", { x: ML + 0.2, y: 2.7, w: 10, h: 1.0, color: WHITE, bold: true, fontFace: F, fontSize: 48 });
s.addText("Q & A", { x: ML + 0.2, y: 3.85, w: 6, h: 0.6, color: CYAN, bold: true, fontFace: F, fontSize: 22, charSpacing: 4 });
s.addText("강화학습 기반 포털 WAS 자원 동시 튜닝  ·  김완수", { x: ML + 0.2, y: 4.7, w: 10, h: 0.4, color: ICE, fontFace: F, fontSize: 13 });

p.writeFile({ fileName: "portal_pool_rl.pptx" }).then(f => console.log("WROTE", f));
