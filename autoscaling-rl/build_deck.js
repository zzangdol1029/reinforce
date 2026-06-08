/* 발표 PPT 생성 스크립트 (pptxgenjs)
 * 사용: npm install pptxgenjs && node build_deck.js
 * results/ 그래프가 없으면 자리표시 박스로 생성된다 (실행 후 재생성 권장).
 */
const pptxgen = require("pptxgenjs");
const path = require("path");

const RES = path.join(__dirname, "results");

// ---- 결과 그림 점검: 없으면 해당 위치에 자리표시 박스를 넣는다 -----------
const fs = require("fs");
const REQUIRED = [
  "traffic_sample.png", "capacity_curve.png",
  "threadpool_learning_curves.png", "threadpool_behavior.png",
  "threadpool_comparison.png",
];
const missing = REQUIRED.filter(f => !fs.existsSync(path.join(RES, f)));
if (missing.length) {
  console.warn("주의: results/ 에 없는 그림은 자리표시로 대체됩니다:");
  missing.forEach(f => console.warn("  - " + f));
  console.warn("python run_all.py 실행 후 node build_deck.js 로 재생성하세요.\n");
}


// ----------------------------------------------------------------- palette
const NAVY = "1E2761";
const NAVY2 = "2E3A75";
const ICE = "CADCFC";
const ORANGE = "C55A11";
const INK = "222222";
const GRAY = "777777";
const LIGHT = "F4F6FB";
const CODE_BG = "F2F2F2";
const GREEN = "2C7A2C";

const HFONT = "Malgun Gothic";
const BFONT = "Malgun Gothic";
const CFONT = "Consolas";

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.author = "김완수";
pres.title = "강화학습 기반 Worker Thread Pool 동적 튜닝";

// ----------------------------------------------------------------- helpers
let pageNo = 0;
function content(title, tag) {
  const s = pres.addSlide();
  s.background = { color: "FFFFFF" };
  pageNo += 1;
  s.addText(title, {
    x: 0.45, y: 0.22, w: 8.0, h: 0.55, fontSize: 22, bold: true,
    color: NAVY, fontFace: HFONT, margin: 0,
  });
  if (tag) {
    s.addShape(pres.shapes.RECTANGLE, { x: 8.6, y: 0.28, w: 1.05, h: 0.34, fill: { color: NAVY } });
    s.addText(tag, { x: 8.6, y: 0.28, w: 1.05, h: 0.34, fontSize: 10, bold: true,
      color: "FFFFFF", fontFace: HFONT, align: "center", valign: "middle", margin: 0 });
  }
  s.addText(String(pageNo + 1), { x: 9.45, y: 5.28, w: 0.4, h: 0.3, fontSize: 10,
    color: GRAY, fontFace: BFONT, align: "right", margin: 0 });
  return s;
}

function bullets(slide, items, opt) {
  const runs = items.map((it, i) => ({
    text: it.t,
    options: {
      bullet: it.lv ? { indent: 10 } : true,
      indentLevel: it.lv || 0,
      bold: !!it.b,
      color: it.c || INK,
      breakLine: true,
      paraSpaceAfter: it.sp != null ? it.sp : 6,
      fontSize: it.fs || opt.fontSize || 13,
    },
  }));
  slide.addText(runs, { fontFace: BFONT, valign: "top", margin: 0, ...opt });
}

function codeBox(slide, code, opt) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x: opt.x, y: opt.y, w: opt.w, h: opt.h, fill: { color: CODE_BG },
    line: { color: "DDDDDD", width: 0.75 },
  });
  slide.addText(
    code.split("\n").map((l) => ({ text: l, options: { breakLine: true } })),
    { x: opt.x + 0.12, y: opt.y + 0.08, w: opt.w - 0.24, h: opt.h - 0.16,
      fontSize: opt.fs || 10, fontFace: CFONT, color: "1A1A1A",
      valign: "top", margin: 0, paraSpaceAfter: 1 }
  );
}

function kpi(slide, x, y, w, big, label, color) {
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h: 1.15, fill: { color: LIGHT } });
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w: 0.07, h: 1.15, fill: { color } });
  slide.addText(big, { x: x + 0.12, y: y + 0.08, w: w - 0.2, h: 0.62, fontSize: 26,
    bold: true, color, fontFace: HFONT, margin: 0 });
  slide.addText(label, { x: x + 0.12, y: y + 0.68, w: w - 0.2, h: 0.42, fontSize: 10.5,
    color: GRAY, fontFace: BFONT, margin: 0 });
}


// 그림 삽입: 파일이 있으면 이미지, 없으면 점선 자리표시 박스
function addImg(slide, file, opt, label) {
  const p = path.join(RES, file);
  if (fs.existsSync(p)) { slide.addImage({ path: p, x: opt.x, y: opt.y, w: opt.w, h: opt.h }); return; }
  slide.addShape(pres.shapes.RECTANGLE, { x: opt.x, y: opt.y, w: opt.w, h: opt.h,
    fill: { color: "FAFAFA" }, line: { color: ORANGE, width: 1, dashType: "dash" } });
  slide.addText("[ " + (label || file) + " ]", { x: opt.x, y: opt.y + opt.h / 2 - 0.35,
    w: opt.w, h: 0.35, fontSize: 12, bold: true, color: ORANGE, fontFace: HFONT,
    align: "center", margin: 0 });
  slide.addText("results/" + file + " 생성 후 node build_deck.js 재실행",
    { x: opt.x, y: opt.y + opt.h / 2 + 0.05, w: opt.w, h: 0.3, fontSize: 9.5,
      color: GRAY, fontFace: BFONT, align: "center", margin: 0 });
}

// ================================================================ 1. 표지
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 3.62, w: 10, h: 0.035, fill: { color: ORANGE } });
  s.addText("강화학습 프로젝트 발표", { x: 0.7, y: 0.9, w: 8.6, h: 0.4, fontSize: 14,
    color: ICE, fontFace: HFONT, margin: 0 });
  s.addText("강화학습 기반 Worker Thread Pool 동적 튜닝", {
    x: 0.7, y: 1.45, w: 8.6, h: 0.85, fontSize: 30, bold: true, color: "FFFFFF",
    fontFace: HFONT, margin: 0 });
  s.addText("인스턴스가 고정된 운영 환경에서 트래픽 변화에 따라 worker thread pool을\n동적으로 튜닝하는 정책을 DQN · PPO로 학습하고 룰 기반 정책과 비교", {
    x: 0.7, y: 2.45, w: 8.6, h: 0.9, fontSize: 15, color: ICE, fontFace: BFONT, margin: 0 });
  s.addText([
    { text: "김완수  ·  충북대학교 산업인공지능학과", options: { breakLine: true, fontSize: 14, bold: true } },
    { text: "2026. 6. 17", options: { fontSize: 12 } },
  ], { x: 0.7, y: 4.0, w: 8.6, h: 0.8, color: "FFFFFF", fontFace: BFONT, margin: 0 });
}

// ================================================================ 2. 목차
{
  const s = content("목차", null);
  const items = [
    ["01", "적용 문제", "고정 인스턴스 제약과 worker thread 튜닝, RL 적용 동기"],
    ["02", "모델 정의", "큐잉 시뮬레이터 (v1 공식/v2 이산사건), MDP 정의"],
    ["03", "방법 / 알고리즘", "DQN(Double DQN), PPO, 룰 기반 베이스라인"],
    ["04", "프로그램 소스", "구조, 환경 · 에이전트 핵심 코드"],
    ["05", "결과 / 분석", "학습 곡선, 정책 거동, 정책별 종합 비교"],
    ["06", "기여 / 향후 연구", "기여도, 한계, 실제 시스템 적용 방향"],
  ];
  items.forEach(([no, t, d], i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.55 + col * 4.75, y = 1.15 + row * 1.35;
    s.addText(no, { x, y, w: 0.75, h: 0.8, fontSize: 30, bold: true, color: ICE,
      fontFace: HFONT, margin: 0 });
    s.addText(t, { x: x + 0.78, y: y + 0.02, w: 3.7, h: 0.4, fontSize: 16, bold: true,
      color: NAVY, fontFace: HFONT, margin: 0 });
    s.addText(d, { x: x + 0.78, y: y + 0.44, w: 3.8, h: 0.62, fontSize: 11, color: GRAY,
      fontFace: BFONT, margin: 0 });
  });
}

// ================================================== 3. 적용 문제 — 배경
{
  const s = content("적용 문제 — MSA 운영에서 자원은 '항상' 어긋난다", "적용 문제");
  bullets(s, [
    { t: "MSA 서비스의 트래픽은 하루 주기로 출렁이고(오전/저녁 피크), 예고 없는 버스트가 끼어든다", b: true },
    { t: "그런데 운영 현장의 worker thread pool 크기는 한 번 튜닝한 고정값", lv: 1 },
    { t: "자원이 부족하면 → 요청이 큐에 쌓여 응답 지연 폭증, SLA 위반", lv: 1 },
    { t: "thread가 남으면 → 메모리(스택)·컨텍스트 스위칭 낭비", lv: 1 },
    { t: "두 비용은 트레이드오프 관계 — '항상 넉넉히'는 답이 아니다", b: true, c: ORANGE },
  ], { x: 0.55, y: 0.95, w: 8.9, h: 2.0, fontSize: 13.5 });
  kpi(s, 0.55, 3.15, 2.9, "지연 ↑", "부족: 큐 대기 → SLA 위반 → 사용자 이탈", ORANGE);
  kpi(s, 3.6, 3.15, 2.9, "비용 ↑", "과잉: 스레드 메모리 · 컨텍스트 스위칭", NAVY);
  kpi(s, 6.65, 3.15, 2.9, "최적점 이동", "트래픽에 따라 적정 자원량이 계속 변함", GREEN);
  s.addText("→ 트래픽을 보면서 자원을 실시간으로 조절하는 '정책'이 필요하다",
    { x: 0.55, y: 4.55, w: 8.9, h: 0.5, fontSize: 15, bold: true, color: NAVY,
      fontFace: HFONT, margin: 0 });
}

// ================================================== 4. 왜 강화학습인가
{
  const s = content("적용 문제 — 왜 강화학습인가", "적용 문제");
  bullets(s, [
    { t: "순차적 의사결정 문제", b: true },
    { t: "지금의 thread 조절 결정이 미래 상태(backlog 누적)에 영향 → MDP 구조", lv: 1 },
    { t: "지연된 보상", b: true },
    { t: "대응이 늦으면 backlog가 누적되어 한동안 지연이 지속 — 피크 전 선제 대응이 이득", lv: 1 },
    { t: "움직이는 최적점", b: true },
    { t: "적정 자원량이 트래픽·워크로드에 따라 계속 변함 → 고정 룰로는 한 점만 커버", lv: 1 },
    { t: "보상 함수로 운영 목표를 직접 표현", b: true },
    { t: "기본 확보량(c_base)·SLA·증설 비용의 가중치로 '운영 정책의 성격'을 설계 가능", lv: 1 },
  ], { x: 0.55, y: 0.95, w: 5.45, h: 4.3, fontSize: 13 });
  s.addShape(pres.shapes.RECTANGLE, { x: 6.1, y: 1.1, w: 3.55, h: 3.6, fill: { color: LIGHT } });
  s.addText("문제 ↔ MDP 대응", { x: 6.3, y: 1.25, w: 3.2, h: 0.35, fontSize: 13, bold: true,
    color: NAVY, fontFace: HFONT, margin: 0 });
  bullets(s, [
    { t: "환경 = MSA 시스템(큐잉 모델)", fs: 11.5, sp: 8 },
    { t: "상태 = 트래픽·이용률·지연·자원량", fs: 11.5, sp: 8 },
    { t: "행동 = thread −4 / 유지 / +4", fs: 11.5, sp: 8 },
    { t: "보상 = −SLA위반 − 기본값 초과 증설비용", fs: 11.5, sp: 8 },
    { t: "정책 = 동적 thread pool 튜너", fs: 11.5, b: true, c: ORANGE },
  ], { x: 6.3, y: 1.7, w: 3.25, h: 2.8 });
}

// ================================================== 5. 인스턴스 고정 제약
{
  const s = content("적용 문제 — 인스턴스 고정 환경에서 유일한 동적 자원", "적용 문제");
  bullets(s, [
    { t: "운영 환경의 제약: 서비스 인스턴스(JVM 프로세스) 수가 고정", b: true },
    { t: "온프레미스 자원·배포 정책상 수평 확장(scale-out)이 불가 → 인스턴스 오토스케일링류 솔루션 적용 불가", lv: 1 },
    { t: "주어진 인스턴스 안에서 실질적으로 조절 가능한 자원 = Undertow worker thread pool", b: true, c: ORANGE },
    { t: "worker thread: 요청을 실제 처리하는 스레드 (컨트롤러 로직·DB 조회·외부 호출)", lv: 1 },
    { t: "런타임 조정 가능 — 재기동 없이 풀 크기를 바꾸는 조정 통로 구현 가능 (ThreadPoolExecutor.setCorePoolSize)", lv: 1 },
  ], { x: 0.55, y: 0.95, w: 8.9, h: 2.35, fontSize: 13 });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.55, y: 3.28, w: 8.9, h: 1.55, fill: { color: LIGHT } });
  s.addText("Undertow 스레드 구조", { x: 0.75, y: 3.4, w: 8.5, h: 0.32, fontSize: 12.5,
    bold: true, color: NAVY, fontFace: HFONT, margin: 0 });
  bullets(s, [
    { t: "IO thread (코어 수): 네트워크 수락·송수신만 — 조절 대상 아님", fs: 11.5, sp: 3 },
    { t: "worker thread (조절 대상): 부족하면 큐 대기 ↑, 과다하면 컨텍스트 스위칭·메모리 ↑", fs: 11.5, sp: 3 },
    { t: "→ 트래픽에 따라 움직이는 최적 thread 수를 실시간 추적하는 것이 본 프로젝트의 문제", fs: 11.5, b: true, c: ORANGE, sp: 3 },
  ], { x: 0.75, y: 3.78, w: 8.5, h: 1.0 });
}


// ================================================== 6. 시뮬레이터 큐잉 모델
{
  const s = content("모델 — 큐잉 이론 기반 시뮬레이터 (M/M/c 근사)", "모델 정의");
  bullets(s, [
    { t: "RL은 수만 step의 시행착오가 필요 → 실제 서버 대신 큐잉 모델 시뮬레이터에서 학습", b: true },
    { t: "자원 c개, 단위 처리율 μ → 시스템 용량 cap = c·μ,  이용률 ρ = λ / cap", lv: 1 },
    { t: "이용률이 1에 다가가면 대기시간이 비선형 폭증 (큐잉 이론의 핵심 성질)", lv: 1 },
    { t: "과부하(λ > cap)면 처리 못 한 요청이 backlog로 누적 → 다음 step의 지연을 키움", lv: 1 },
    { t: "→ '한 번 밀리면 한동안 느려지는' 실제 서버 거동을 재현", lv: 1, c: ORANGE },
  ], { x: 0.55, y: 0.95, w: 8.9, h: 1.9, fontSize: 13 });
  codeBox(s, [
    "# 응답시간 (autoscale_env.py)",
    "W = 1/μ                          # 순수 처리시간",
    "  + ρ / (cap · (1−ρ))           # 정상상태 큐 대기 (ρ<1)",
    "  + backlog / cap               # 누적 잔여작업 소진 시간",
    "",
    "# backlog 동역학 — 매 step",
    "backlog = max(0, backlog + (λ − cap))",
  ].join("\n"), { x: 0.55, y: 2.85, w: 4.55, h: 2.0, fs: 10.5 });
  addImg(s, "traffic_sample.png", { x: 5.3, y: 2.85, w: 4.3, h: 1.72 }, "트래픽 샘플");
  s.addText("트래픽 생성기: 일일 사이클(오전/저녁 피크) + 랜덤 버스트 — 에피소드마다 패턴이 바뀌어 암기 불가",
    { x: 5.3, y: 4.62, w: 4.3, h: 0.55, fontSize: 9.5, color: GRAY, fontFace: BFONT, margin: 0 });
}

// ================================================== 6b. RL 상호작용 구조도
{
  const s = content("모델 — Agent · Environment 상호작용 구조", "모델 정의");

  // ---------- Agent 박스 (왼쪽) ----------
  s.addShape(pres.shapes.RECTANGLE, { x: 0.55, y: 1.5, w: 3.1, h: 2.5,
    fill: { color: LIGHT }, line: { color: NAVY, width: 1.5 } });
  s.addText("Agent", { x: 0.55, y: 1.62, w: 3.1, h: 0.4, fontSize: 17, bold: true,
    color: NAVY, fontFace: HFONT, align: "center", margin: 0 });
  s.addText("스케일링 정책 (DQN / PPO)", { x: 0.55, y: 2.02, w: 3.1, h: 0.32,
    fontSize: 11.5, bold: true, color: ORANGE, fontFace: HFONT, align: "center", margin: 0 });
  bullets(s, [
    { t: "정책 신경망 MLP (2×128)", fs: 10.5, sp: 3 },
    { t: "입력: 상태 벡터 (관측 변수)", fs: 10.5, sp: 3 },
    { t: "출력: 행동 3개 중 1개 선택", fs: 10.5, sp: 3 },
    { t: "학습: 누적 보상 최대화", fs: 10.5, sp: 3 },
  ], { x: 0.8, y: 2.45, w: 2.7, h: 1.45 });

  // ---------- Environment 박스 (오른쪽) ----------
  s.addShape(pres.shapes.RECTANGLE, { x: 6.35, y: 1.5, w: 3.1, h: 2.5,
    fill: { color: LIGHT }, line: { color: ORANGE, width: 1.5 } });
  s.addText("Environment", { x: 6.35, y: 1.62, w: 3.1, h: 0.4, fontSize: 17, bold: true,
    color: ORANGE, fontFace: HFONT, align: "center", margin: 0 });
  s.addText("MSA 서비스 시뮬레이터", { x: 6.35, y: 2.02, w: 3.1, h: 0.32,
    fontSize: 11.5, bold: true, color: NAVY, fontFace: HFONT, align: "center", margin: 0 });
  bullets(s, [
    { t: "트래픽 생성기 λ(t): 일일 피크+버스트", fs: 10.5, sp: 3 },
    { t: "FCFS 큐: worker thread c개가 처리", fs: 10.5, sp: 3 },
    { t: "비단조 용량·backlog·지연 계산", fs: 10.5, sp: 3 },
    { t: "상태·보상을 만들어 Agent에 반환", fs: 10.5, sp: 3 },
  ], { x: 6.6, y: 2.45, w: 2.75, h: 1.45 });

  // ---------- 행동 화살표 (Agent -> Env, 위) ----------
  s.addShape(pres.shapes.LINE, { x: 3.65, y: 1.95, w: 2.7, h: 0,
    line: { color: NAVY, width: 2.5, endArrowType: "triangle" } });
  s.addText([
    { text: "행동 aₜ", options: { bold: true, color: NAVY, fontSize: 12, breakLine: true } },
    { text: "thread −4  /  유지  /  +4", options: { fontSize: 10.5, color: INK } },
  ], { x: 3.65, y: 1.28, w: 2.7, h: 0.62, fontFace: BFONT, align: "center", margin: 0 });

  // ---------- 상태+보상 화살표 (Env -> Agent, 박스 사이 아래쪽) ----------
  s.addShape(pres.shapes.LINE, { x: 3.65, y: 3.45, w: 2.7, h: 0, flipH: true,
    line: { color: ORANGE, width: 2.5, endArrowType: "triangle" } });
  s.addText([
    { text: "상태 sₜ₊₁  +  보상 rₜ₊₁", options: { bold: true, color: ORANGE, fontSize: 12, breakLine: true } },
    { text: "(환경 관측 변수 + 행동 채점 결과)", options: { fontSize: 10, color: GRAY } },
  ], { x: 3.55, y: 3.55, w: 2.9, h: 0.62, fontFace: BFONT, align: "center", margin: 0 });

  // ---------- 하단: 상태 변수 / 보상 변수 정리 ----------
  s.addShape(pres.shapes.RECTANGLE, { x: 0.55, y: 4.15, w: 4.35, h: 1.25, fill: { color: "FFFFFF" },
    line: { color: ORANGE, width: 1 } });
  s.addText("상태(환경) 변수 — Agent가 매 step 관측", { x: 0.7, y: 4.22, w: 4.1, h: 0.28,
    fontSize: 10.5, bold: true, color: ORANGE, fontFace: HFONT, margin: 0 });
  s.addText("트래픽 λ · λ 변화량(추세) · 이용률 ρ · p95지연/SLA ·\n평균지연/SLA · 현재 thread 수 c · backlog(잔여작업)",
    { x: 0.7, y: 4.52, w: 4.1, h: 0.8, fontSize: 10, color: INK, fontFace: BFONT, margin: 0 });

  s.addShape(pres.shapes.RECTANGLE, { x: 5.1, y: 4.15, w: 4.35, h: 1.25, fill: { color: "FFFFFF" },
    line: { color: NAVY, width: 1 } });
  s.addText("보상 변수 — 행동의 좋고 나쁨을 채점", { x: 5.25, y: 4.22, w: 4.1, h: 0.28,
    fontSize: 10.5, bold: true, color: NAVY, fontFace: HFONT, margin: 0 });
  s.addText("SLA 초과 정도(p95 기준) · 기본값(c_base=16) 초과 증설분 ·\n기본값 미만 정도 · 행동 변경 여부  →  다음 장에서 상세",
    { x: 5.25, y: 4.52, w: 4.1, h: 0.8, fontSize: 10, color: INK, fontFace: BFONT, margin: 0 });
}

// ================================================== 6c. 보상 설계 상세
{
  const s = content("모델 — 보상 설계: 어떤 행동이 보상받는가", "모델 정의");

  // 보상 수식
  s.addShape(pres.shapes.RECTANGLE, { x: 0.55, y: 0.95, w: 8.9, h: 0.62, fill: { color: LIGHT } });
  s.addText([
    { text: "r  =  − 1.0·(p95 SLA 초과비율)", options: { bold: true, color: "C0392B" } },
    { text: "  − 0.30·(c − 16)⁺/48", options: { bold: true, color: NAVY } },
    { text: "  − 0.2·(16 − c)⁺/16", options: { bold: true, color: ORANGE } },
    { text: "  − 0.02·(행동 변경)", options: { bold: true, color: GRAY } },
  ], { x: 0.75, y: 1.05, w: 8.5, h: 0.45, fontSize: 14, fontFace: CFONT, margin: 0 });

  // 상황별 보상 카드 4개
  const cards = [
    ["SLA를 지키며 기본값(16) 유지", "페널티 0 — 가장 좋은 상태",
     "평소 트래픽에서는 증설하지 않고 기본값에 머무는 것이 최선", GREEN, "보상 ◎"],
    ["트래픽 증가 → 선제 증설로 SLA 방어", "증설분만 소액 비용 (−0.30·초과분/48)",
     "SLA 위반 페널티(−1.0×)보다 훨씬 싸다 → 피크엔 증설이 이득", NAVY, "보상 ○"],
    ["대응 실패 → p95가 SLA 초과", "초과 정도에 비례한 큰 페널티 (최대 −5)",
     "보상 체계에서 가장 비싼 실수 — SLA 준수가 최우선임을 학습", "C0392B", "페널티 ✕"],
    ["기본값 미만 감축 / 잦은 변경", "−0.2·미만 정도 / 변경마다 −0.02",
     "안전 여유 훼손과 설정 진동(thrashing)을 억제", ORANGE, "페널티 △"],
  ];
  cards.forEach(([t, r, d, color, badge], i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.55 + col * 4.55, y = 1.85 + row * 1.5;
    s.addShape(pres.shapes.RECTANGLE, { x, y, w: 4.35, h: 1.35, fill: { color: "FFFFFF" },
      line: { color: "DDDDDD", width: 1 } });
    s.addShape(pres.shapes.RECTANGLE, { x, y, w: 0.07, h: 1.35, fill: { color } });
    s.addText(badge, { x: x + 3.3, y: y + 0.08, w: 0.95, h: 0.3, fontSize: 10.5, bold: true,
      color, fontFace: HFONT, align: "right", margin: 0 });
    s.addText(t, { x: x + 0.18, y: y + 0.08, w: 3.2, h: 0.32, fontSize: 11.5, bold: true,
      color: NAVY, fontFace: HFONT, margin: 0 });
    s.addText(r, { x: x + 0.18, y: y + 0.42, w: 4.0, h: 0.3, fontSize: 10.5, bold: true,
      color, fontFace: BFONT, margin: 0 });
    s.addText(d, { x: x + 0.18, y: y + 0.74, w: 4.0, h: 0.55, fontSize: 10, color: GRAY,
      fontFace: BFONT, margin: 0 });
  });

  s.addText("→ 보상이 유도하는 최적 행동 패턴: 평소엔 기본값 16 유지 → 트래픽 상승 조짐에 선제 증설 → 피크가 지나면 기본값 복귀",
    { x: 0.55, y: 4.95, w: 8.9, h: 0.45, fontSize: 13, bold: true, color: NAVY,
      fontFace: HFONT, margin: 0 });
}

// ================================================== 8. MDP Stage 2
{
  const s = content("모델 — MDP 정의 (worker thread 튜닝)", "모델 정의");
  bullets(s, [
    { t: "State (6,): [ λ, λ 변화량, ρ, 지연/SLA, thread 수 c, backlog ]", b: true },
    { t: "Action (3): thread −4 / 유지 / +4", b: true },
    { t: "Reward: −SLA위반 − 0.30·(기본 16 초과분) − 0.2·(미만) − 0.02·행동", b: true },
    { t: "핵심 차이: 용량이 비단조 — thread를 늘릴수록 좋은 게 아니다", b: true, c: ORANGE },
    { t: "c ≤ 32: IO-bound라 병렬화 이득 → 용량 증가", lv: 1 },
    { t: "c > 32: 컨텍스트 스위칭·CPU 경합 → thread당 효율 하락, 용량 감소", lv: 1 },
    { t: "에이전트는 트래픽에 따라 움직이는 최적점을 추적해야 함", lv: 1 },
  ], { x: 0.55, y: 0.95, w: 4.6, h: 3.4, fontSize: 12.5 });
  codeBox(s, [
    "# 비단조 용량 (threadpool_env.py)",
    "overhead = 1 + α·max(0, c−knee)²/knee²",
    "capacity = c·μ_t / overhead",
  ].join("\n"), { x: 0.55, y: 4.35, w: 4.6, h: 1.0, fs: 10.5 });
  addImg(s, "capacity_curve.png", { x: 5.4, y: 1.15, w: 4.2, h: 2.36 }, "비단조 용량 곡선");
  s.addText("실측 곡선: c=37 부근에서 용량 피크 → 그 이상은 역효과.\n룰 기반('이용률 높으면 늘려라')이 함정에 빠질 수 있는 구조",
    { x: 5.4, y: 3.6, w: 4.2, h: 0.8, fontSize: 10.5, color: GRAY, fontFace: BFONT, margin: 0 });
}


// ================================================== 8b. v2 이산사건 환경
{
  const s = content("모델 — v2 이산사건 시뮬레이터 (충실도 향상)", "모델 정의");
  s.addText("같은 MDP 구조에서 시뮬레이션 충실도를 올린 v2 환경을 추가 — 요청 단위 시뮬레이션으로 p95(tail latency) 기반 SLA 평가",
    { x: 0.55, y: 0.9, w: 8.9, h: 0.55, fontSize: 13, color: INK, fontFace: BFONT, margin: 0 });
  const rows = [
    [
      { text: "구분", options: { bold: true, color: "FFFFFF", fill: { color: NAVY }, align: "center" } },
      { text: "v1 — 큐잉 공식", options: { bold: true, color: "FFFFFF", fill: { color: NAVY }, align: "center" } },
      { text: "v2 — 이산사건 (event-driven)", options: { bold: true, color: "FFFFFF", fill: { color: ORANGE }, align: "center" } },
    ],
    [{ text: "지연 계산", options: { bold: true, fill: { color: LIGHT } } },
      "M/M/c 공식으로 평균 근사", "요청 하나하나를 FCFS 큐에 배정해 정확히 계산"],
    [{ text: "SLA 지표", options: { bold: true, fill: { color: LIGHT } } },
      "평균 지연", "p95 (tail latency) — 실무 SLA와 동일 기준"],
    [{ text: "처리시간", options: { bold: true, fill: { color: LIGHT } } },
      "상수 (1/μ)", "lognormal heavy-tail — 가끔 오는 무거운 요청 재현"],
    [{ text: "도메인 랜덤화", options: { bold: true, fill: { color: LIGHT } } },
      "트래픽 패턴", "트래픽 패턴 + tail 강도(σ) — 과적합 방지 강화"],
    [{ text: "용도", options: { bold: true, fill: { color: LIGHT } } },
      "빠른 반복 실험 (1 ep ≈ 0.01s)", "충실도 높은 최종 평가 (1 ep ≈ 0.05~0.1s)"],
  ];
  s.addTable(rows, { x: 0.55, y: 1.5, w: 8.9, colW: [1.5, 3.2, 4.2], fontSize: 11.5,
    fontFace: BFONT, color: INK, border: { pt: 0.75, color: "CCCCCC" },
    valign: "middle", rowH: 0.5 });
  bullets(s, [
    { t: "스케일 결정·cold start·비단조 용량 등 문제 구조는 v1과 동일 — 같은 에이전트 코드로 학습", b: true },
    { t: "scale-in은 drain 방식: 처리 중인 요청을 끊지 않고 한가한 워커부터 제거 (실제 운영과 동일)", lv: 1 },
  ], { x: 0.55, y: 4.6, w: 8.9, h: 0.8, fontSize: 12 });
}

// ================================================== 9. DQN
{
  const s = content("알고리즘 ① DQN (Double DQN)", "방법");
  bullets(s, [
    { t: "가치 기반(off-policy): 상태별 행동가치 Q(s,a)를 학습, 행동은 argmax Q", b: true },
    { t: "Q 네트워크: MLP (state → 64 → 64 → 3행동), DeZero로 직접 구현", lv: 1 },
    { t: "Experience Replay (5만 transition)", b: true },
    { t: "시계열 상관 제거 + 과거의 '위기 상황(backlog 폭증)'을 반복 재학습", lv: 1 },
    { t: "Target Network (500 step마다 동기화)", b: true },
    { t: "타깃 안정화로 발산 방지", lv: 1 },
    { t: "Double DQN", b: true },
    { t: "행동 선택은 online, 평가는 target net → Q값 과대추정 완화", lv: 1 },
    { t: "ε-greedy: 1.0 → 0.05 선형 감쇠 (학습의 60% 지점까지)", b: true },
  ], { x: 0.55, y: 0.95, w: 5.3, h: 4.3, fontSize: 12.5 });
  codeBox(s, [
    "# Double DQN 타깃 (agents/dqn.py)",
    "a_next = argmax_a Q_online(s', a)",
    "y = r + γ·(1−done)·Q_target(s', a_next)",
    "loss = mean( (Q(s,a) − y)² )",
  ].join("\n"), { x: 6.1, y: 1.3, w: 3.5, h: 1.5, fs: 10 });
  s.addShape(pres.shapes.RECTANGLE, { x: 6.1, y: 3.1, w: 3.5, h: 1.7, fill: { color: LIGHT } });
  s.addText("왜 이 문제에 맞나", { x: 6.25, y: 3.22, w: 3.2, h: 0.3, fontSize: 12, bold: true,
    color: NAVY, fontFace: HFONT, margin: 0 });
  s.addText("행동이 3개뿐인 이산 공간 + replay로 드문 과부하 상태를 잊지 않음 → 강건한 정책",
    { x: 6.25, y: 3.55, w: 3.2, h: 1.1, fontSize: 11, color: INK, fontFace: BFONT, margin: 0 });
}

// ================================================== 10. PPO
{
  const s = content("알고리즘 ② PPO (Clipped Surrogate)", "방법");
  bullets(s, [
    { t: "정책 경사(on-policy): 정책 π(a|s)를 직접 최적화", b: true },
    { t: "Actor(정책)·Critic(가치) 분리 MLP, DeZero로 직접 구현", lv: 1 },
    { t: "GAE(λ=0.95): advantage 추정의 분산 감소", b: true },
    { t: "Clipped objective: 정책 갱신 비율을 [1−ε, 1+ε]로 제한 (ε=0.2)", b: true },
    { t: "한 번의 업데이트로 정책이 급변하는 것을 방지 → 안정적 학습", lv: 1 },
    { t: "Entropy bonus(0.01): 조기 수렴 방지", b: true },
    { t: "rollout 2048 step마다 10 epoch 미니배치 학습", b: true },
  ], { x: 0.55, y: 0.95, w: 5.3, h: 3.6, fontSize: 12.5 });
  codeBox(s, [
    "# PPO 손실 (agents/ppo.py)",
    "ratio = exp(logπ(a|s) − logπ_old(a|s))",
    "L_clip = min( ratio·A,",
    "              clip(ratio,1−ε,1+ε)·A )",
    "loss = −L_clip + 0.5·L_value − 0.01·H(π)",
  ].join("\n"), { x: 6.1, y: 1.3, w: 3.5, h: 1.75, fs: 10 });
  s.addShape(pres.shapes.RECTANGLE, { x: 6.1, y: 3.35, w: 3.5, h: 1.45, fill: { color: LIGHT } });
  s.addText("DQN과의 비교 관전 포인트", { x: 6.25, y: 3.47, w: 3.2, h: 0.3, fontSize: 12,
    bold: true, color: NAVY, fontFace: HFONT, margin: 0 });
  s.addText("확률적 정책·on-policy 학습이 ① 점진 트래픽 변화, ② 드문 위기 상태에서 각각 어떻게 다르게 반응하는가",
    { x: 6.25, y: 3.78, w: 3.2, h: 0.95, fontSize: 11, color: INK, fontFace: BFONT, margin: 0 });
}

// ================================================== 11. 베이스라인
{
  const s = content("비교 베이스라인 — 룰 기반 · 고정 설정", "방법");
  s.addShape(pres.shapes.RECTANGLE, { x: 0.55, y: 1.05, w: 4.3, h: 3.0, fill: { color: LIGHT } });
  s.addText("Rule-based (HPA식 임계값)", { x: 0.75, y: 1.2, w: 3.9, h: 0.35, fontSize: 14,
    bold: true, color: NAVY, fontFace: HFONT, margin: 0 });
  bullets(s, [
    { t: "이용률 ρ > 0.7 → scale-out", fs: 12 },
    { t: "이용률 ρ < 0.3 → scale-in", fs: 12 },
    { t: "쿨다운 3 step (진동 방지)", fs: 12 },
    { t: "Kubernetes HPA가 실제로 쓰는 방식 — 실무 대표 베이스라인", fs: 12, b: true },
  ], { x: 0.75, y: 1.65, w: 3.9, h: 2.3 });
  s.addShape(pres.shapes.RECTANGLE, { x: 5.15, y: 1.05, w: 4.3, h: 3.0, fill: { color: LIGHT } });
  s.addText("Static (고정 설정)", { x: 5.35, y: 1.2, w: 3.9, h: 0.35, fontSize: 14,
    bold: true, color: NAVY, fontFace: HFONT, margin: 0 });
  bullets(s, [
    { t: "에피소드 내내 thread 16개(기본값) 고정", fs: 12 },
    { t: "인스턴스 수는 모든 정책에서 고정 — 운영 제약", fs: 12 },
    { t: "'튜닝 한 번 하고 방치'하는 현장 관행의 대조군", fs: 12, b: true },
  ], { x: 5.35, y: 1.65, w: 3.9, h: 2.3 });
  s.addText("평가 프로토콜: 학습에 쓰지 않은 고정 시드 트래픽 20 에피소드에서 4개 정책을 동일 조건 비교",
    { x: 0.55, y: 4.45, w: 8.9, h: 0.45, fontSize: 13, bold: true, color: NAVY,
      fontFace: HFONT, margin: 0 });
}

// ================================================== 12. 소스 구조
{
  const s = content("프로그램 소스 — 구조", "소스");
  codeBox(s, [
    "autoscaling-rl/",
    "├── config.py              # 환경·알고리즘 하이퍼파라미터",
    "├── envs/",
    "│   ├── traffic.py         # 일일 패턴+버스트 트래픽 생성기",
    "│   ├── threadpool_env.py  # v1: thread 환경 (큐잉 공식)",
    "│   ├── event_env.py       # v2: 이산사건 환경 (p95 SLA)",
    "│   └── autoscale_env.py   # (확장용) 인스턴스 환경",
    "├── agents/",
    "│   ├── dqn.py             # Double DQN (직접 구현)",
    "│   └── ppo.py             # PPO + GAE (직접 구현)",
    "├── baselines.py           # HPA식 룰 기반·고정 정책",
    "├── train.py               # 학습 (체크포인트 재개 지원)",
    "├── evaluate.py            # 평가 + 그래프 생성",
    "└── results/               # 곡선·모델·그래프·요약 csv",
  ].join("\n"), { x: 0.55, y: 1.0, w: 5.1, h: 3.9, fs: 10.5 });
  bullets(s, [
    { t: "딥러닝 라이브러리: 교재의 DeZero 사용", b: true },
    { t: "MLP·옵티마이저만 라이브러리, DQN/PPO 알고리즘 로직은 전부 직접 구현", lv: 1 },
    { t: "환경은 Gymnasium 표준 인터페이스", b: true },
    { t: "reset() / step() → 다른 RL 라이브러리로도 즉시 교체 검증 가능", lv: 1 },
    { t: "재현 가능성", b: true },
    { t: "시드 고정, 평가 트래픽 분리, requirements.txt 제공", lv: 1 },
  ], { x: 5.95, y: 1.1, w: 3.7, h: 3.8, fontSize: 12 });
}

// ================================================== 13. 소스 — 환경
{
  const s = content("프로그램 소스 — 환경 핵심 (step 함수)", "소스");
  codeBox(s, [
    "def capacity(self, c):",
    "    # 비단조 용량: knee 초과 시 컨텍스트 스위칭 오버헤드",
    "    overhead = 1 + alpha * max(0, c - knee)**2 / knee**2",
    "    return c * mu_thread / overhead",
    "",
    "def step(self, action):",
    "    delta = (int(action) - 1) * self.c_step   # -4 / 0 / +4",
    "    self.c = clip(self.c + delta, c_min, c_max)",
    "",
    "    # 큐 동역학: 못 받아낸 요청은 backlog로 누적",
    "    cap = self.capacity(self.c)",
    "    self.backlog = max(0, self.backlog + (lam - cap))",
    "    self.latency = self._latency(lam)",
    "",
    "    # 보상: SLA 위반 + 기본값(c_base) 초과 비용 + 행동 비용",
    "    sla_excess = max(0, (latency - sla) / sla)",
    "    cost  = max(0, c - c_base) / (c_max - c_base)",
    "    under = max(0, c_base - c) / c_base",
    "    r = -w_sla*sla_excess - w_cost*cost \\",
    "        -w_under*under - w_thrash*|delta|/c_step",
  ].join("\n"), { x: 0.55, y: 0.95, w: 5.6, h: 4.3, fs: 9 });
  bullets(s, [
    { t: "행동 → 상태 전이 → 보상의 전 과정이 20줄 내외", b: true },
    { t: "비단조 용량: thread를 늘리는 행동이 항상 이득이 아님 — 환경이 '함정'을 가짐", lv: 1 },
    { t: "backlog: 과부하의 여파가 다음 step으로 이어짐 — state에 포함해 MDP 유지", lv: 1 },
    { t: "c_base 기준 비용: 평소엔 기본값 유지, 피크에만 증설하는 정책을 유도", lv: 1 },
  ], { x: 6.35, y: 1.2, w: 3.3, h: 4.0, fontSize: 11.5 });
}


// ================================================== 14. 소스 — DQN
{
  const s = content("프로그램 소스 — DQN 학습 루프", "소스");
  codeBox(s, [
    "def learn(self):",
    "    s, a, r, s2, d = self.buffer.sample(64)",
    "",
    "    # Double DQN: 선택은 online, 평가는 target",
    "    a_next = self.q(s2).data.argmax(axis=1)",
    "    q_next = self.target(s2).data[arange(B), a_next]",
    "    y = r + gamma * (1 - d) * q_next",
    "",
    "    q_sa = F.get_item(self.q(s), (arange(B), a))",
    "    loss = F.sum((q_sa - y) ** 2) / B",
    "",
    "    self.q.cleargrads()",
    "    loss.backward()        # DeZero 자동미분",
    "    clip_grads([self.q], 10.0)",
    "    self.opt.update()      # Adam",
    "",
    "    if self.step_count % 500 == 0:",
    "        self.target = copy.deepcopy(self.q)",
  ].join("\n"), { x: 0.55, y: 1.0, w: 5.6, h: 4.3, fs: 9.5 });
  bullets(s, [
    { t: "타깃 y는 기울기 차단(.data) — 학습 대상은 Q(s,a)만", b: true },
    { t: "Double DQN 한 줄 차이: a_next를 online net에서 선택", lv: 1 },
    { t: "global-norm gradient clipping은 DeZero에 없어 직접 구현", lv: 1 },
    { t: "ε-greedy·replay·target sync 주기는 config.py에서 일괄 관리", lv: 1 },
  ], { x: 6.35, y: 1.2, w: 3.3, h: 3.6, fontSize: 11.5 });
}

// ================================================== 15. 소스 — PPO
{
  const s = content("프로그램 소스 — PPO 업데이트", "소스");
  codeBox(s, [
    "# GAE(λ)로 advantage 계산",
    "for t in reversed(range(T)):",
    "    delta = r[t] + γ·V[t+1]·(1−done) − V[t]",
    "    gae   = delta + γ·λ·(1−done)·gae",
    "    adv[t] = gae",
    "adv = (adv − adv.mean()) / adv.std()",
    "",
    "# Clipped surrogate (미니배치 × 10 epoch)",
    "ratio = F.exp(logp − logp_old)",
    "surr1 = ratio * adv",
    "surr2 = F.clip(ratio, 1−ε, 1+ε) * adv",
    "take1 = (surr1.data <= surr2.data)   # min 선택 마스크",
    "pi_loss = -F.sum(surr1*take1 + surr2*(1−take1)) / m",
    "",
    "loss = pi_loss + 0.5*v_loss − 0.01*entropy",
  ].join("\n"), { x: 0.55, y: 1.0, w: 5.6, h: 4.0, fs: 9.5 });
  bullets(s, [
    { t: "DeZero에 element-wise min이 없어 선택 마스크로 직접 구현", b: true },
    { t: "min의 기울기는 '작은 쪽으로만 흐른다'와 수학적으로 동일", lv: 1 },
    { t: "advantage 정규화로 학습 안정화", lv: 1 },
    { t: "actor·critic 분리 옵티마이저 — 가치 손실이 정책을 흔들지 않도록", lv: 1 },
  ], { x: 6.35, y: 1.2, w: 3.3, h: 3.6, fontSize: 11.5 });
}

// ================================================== 16. 실험 설정
{
  const s = content("실험 설정", "실험");
  const rows = [
    [
      { text: "구분", options: { bold: true, color: "FFFFFF", fill: { color: NAVY }, align: "center" } },
      { text: "값", options: { bold: true, color: "FFFFFF", fill: { color: NAVY }, align: "center" } },
    ],
    [{ text: "학습량", options: { bold: true, fill: { color: LIGHT } } },
      "v1(threadpool): DQN 500, PPO 600 / v2(ev-threadpool): DQN 200, PPO 300 에피소드 권장 (1 ep = 288 step)"],
    [{ text: "DQN", options: { bold: true, fill: { color: LIGHT } } },
      "hidden 128, lr 5e-4, buffer 100k, batch 128, target sync 1000 step, ε 1.0→0.05 (학습 50%까지 감쇠)"],
    [{ text: "PPO", options: { bold: true, fill: { color: LIGHT } } },
      "hidden 128, lr 3e-4, GAE λ 0.95, clip 0.2, rollout 4096, 10 epoch, minibatch 256, entropy 0.01"],
    [{ text: "네트워크", options: { bold: true, fill: { color: LIGHT } } },
      "MLP 2×128 (DQN: ReLU / PPO: tanh), DeZero 구현 — 권장값 근거는 config.py 주석"],
    [{ text: "평가", options: { bold: true, fill: { color: LIGHT } } },
      "평가 전용 시드 20 에피소드(학습·검증 미사용), greedy, 검증 best 가중치 사용"],
    [{ text: "지표", options: { bold: true, fill: { color: LIGHT } } },
      "에피소드 보상 / SLA 위반률(step 비율) / 평균 자원 비용 / 평균 지연"],
  ];
  s.addTable(rows, { x: 0.55, y: 1.0, w: 8.9, colW: [1.6, 7.3], fontSize: 11.5,
    fontFace: BFONT, color: INK, border: { pt: 0.75, color: "CCCCCC" },
    valign: "middle", rowH: 0.52 });
  s.addText("컴퓨팅: CPU만으로 충분 (v1 1~2분, v2 3~5분 / 알고리즘당) — hp_search.py로 환경별 최적값 탐색 가능",
    { x: 0.55, y: 4.85, w: 8.9, h: 0.45, fontSize: 12, color: GRAY, fontFace: BFONT, margin: 0 });
}


// ================================================== 16b. 과적합 방지
{
  const s = content("과적합 방지 설계", "실험");
  const items = [
    ["1", "시드 3분할 — 학습 / 검증 / 평가 트래픽 완전 분리",
      "학습(seed×10000+ep), 검증(VAL_SEED), 최종 평가(EVAL_SEED)가 절대 겹치지 않음. 평가 시드는 하이퍼파라미터 튜닝에도 사용하지 않는다 — 지도학습의 train/val/test 분리와 동일한 원칙."],
    ["2", "검증 기반 best 체크포인트",
      "10 에피소드마다 검증 트래픽에서 greedy 평가 → 최고 성능 시점의 가중치를 별도 저장. 학습 후반의 정책 붕괴(v1 PPO에서 관찰)가 최종 모델을 오염시키지 않는다. 최종 평가는 항상 best 가중치로 수행."],
    ["3", "도메인 랜덤화",
      "트래픽 위상·피크 크기·버스트 위치 + (v2) heavy-tail 강도 σ가 에피소드마다 무작위 — 특정 패턴을 암기하는 정책은 검증에서 걸러진다."],
  ];
  items.forEach(([no, t, d], i) => {
    const y = 1.0 + i * 1.42;
    s.addShape(pres.shapes.RECTANGLE, { x: 0.55, y, w: 8.9, h: 1.26, fill: { color: LIGHT } });
    s.addShape(pres.shapes.RECTANGLE, { x: 0.55, y, w: 0.07, h: 1.26, fill: { color: NAVY } });
    s.addText(no, { x: 0.75, y: y + 0.18, w: 0.6, h: 0.9, fontSize: 30, bold: true,
      color: ICE, fontFace: HFONT, margin: 0 });
    s.addText(t, { x: 1.45, y: y + 0.1, w: 7.85, h: 0.35, fontSize: 13.5, bold: true,
      color: NAVY, fontFace: HFONT, margin: 0 });
    s.addText(d, { x: 1.45, y: y + 0.48, w: 7.85, h: 0.72, fontSize: 10.5, color: INK,
      fontFace: BFONT, margin: 0 });
  });
}

// ================================================== 17. 결과 ① v1 학습곡선
{
  const s = content("결과 ① v1 환경 — 학습 곡선", "결과");
  addImg(s, "threadpool_learning_curves.png", { x: 0.55, y: 1.0, w: 6.2, h: 3.6 }, "v1 학습 곡선");
  bullets(s, [
    { t: "실선: 학습 보상(10-ep 이동평균) / 점선: 검증 트래픽 보상", b: true, fs: 11.5 },
    { t: "검증 곡선의 최고점에서 best 체크포인트가 저장됨 — 최종 평가에 사용", fs: 11.5 },
    { t: "관전 포인트: 무작위 초기 정책 대비 수렴 수준, DQN(ε 감쇠를 따라 점진 개선) vs PPO(빠른 수렴) 형태 차이", fs: 11.5 },
    { t: "수렴 후 출렁임은 에피소드마다 트래픽 난이도(버스트 수·크기)가 달라서 — 비정상 아님", fs: 11.5 },
  ], { x: 6.95, y: 1.2, w: 2.75, h: 3.9, fontSize: 11.5 });
}

// ================================================== 18. 결과 ② v1 거동
{
  const s = content("결과 ② v1 환경 — 같은 트래픽에서의 정책 거동", "결과");
  addImg(s, "threadpool_behavior.png", { x: 0.45, y: 0.92, w: 5.55, h: 4.44 }, "v1 정책 거동");
  bullets(s, [
    { t: "위: 트래픽 / 가운데: worker thread 수 / 아래: 응답 지연(빨간 점선 = SLA)", fs: 11 },
    { t: "확인 포인트 ①: RL 정책이 평소 기본값(16) 부근을 유지하다 피크에 증설 후 복귀하는가", b: true },
    { t: "확인 포인트 ②: thread를 무작정 늘리지 않는가 — 비단조 용량의 함정 회피", lv: 1 },
    { t: "확인 포인트 ③: Rule-based의 사후 대응 지연 스파이크 vs RL의 선제 대응", lv: 1 },
    { t: "Static(회색)은 버스트마다 SLA 붕괴 → 동적 제어의 필요성 입증", lv: 1 },
  ], { x: 6.15, y: 1.2, w: 3.5, h: 3.9, fontSize: 11.5 });
}

// ================================================== 19. 결과 ③ v1 비교
{
  const s = content("결과 ③ v1 환경 — 정책 종합 비교", "결과");
  const rows = [
    [
      { text: "정책", options: { bold: true, color: "FFFFFF", fill: { color: NAVY }, align: "center" } },
      { text: "보상", options: { bold: true, color: "FFFFFF", fill: { color: NAVY }, align: "center" } },
      { text: "SLA 위반률", options: { bold: true, color: "FFFFFF", fill: { color: NAVY }, align: "center" } },
      { text: "자원 비용", options: { bold: true, color: "FFFFFF", fill: { color: NAVY }, align: "center" } },
    ],
    [{ text: "DQN", options: { bold: true } }, "(기입)", "(기입)", "(기입)"],
    [{ text: "PPO", options: { bold: true } }, "(기입)", "(기입)", "(기입)"],
    [{ text: "Rule-based", options: { bold: true } }, "(기입)", "(기입)", "(기입)"],
    [{ text: "Static", options: { color: GRAY } }, "(기입)", "(기입)", "(기입)"],
  ];
  s.addText("evaluate.py 출력(results/threadpool_summary.csv)을 그대로 기입 — 비용은 c_base 초과분 기준",
    { x: 0.55, y: 0.92, w: 8.9, h: 0.4, fontSize: 12, color: GRAY, fontFace: BFONT, margin: 0 });
  s.addTable(rows, { x: 0.55, y: 1.45, w: 5.4, colW: [1.4, 1.3, 1.4, 1.3], fontSize: 11.5,
    fontFace: BFONT, color: INK, border: { pt: 0.75, color: "CCCCCC" },
    align: "center", valign: "middle", rowH: 0.42 });
  bullets(s, [
    { t: "해석 가이드", b: true },
    { t: "RL이 룰 기반 대비 같은 비용으로 SLA 위반을 줄이거나, 같은 위반률을 더 적은 증설로 달성하면 우위", lv: 1 },
    { t: "DQN과 PPO가 다른 운영점(안전形 vs 효율形)에 수렴하는지 확인", lv: 1 },
  ], { x: 6.15, y: 1.5, w: 3.5, h: 2.1, fontSize: 11.5 });
  addImg(s, "threadpool_comparison.png", { x: 0.85, y: 3.55, w: 5.85, h: 1.85 }, "정책 비교 그래프");
}


// ================================================== 21b. v2 결과 placeholder
{
  const s = content("결과 ⑥ v2 이산사건 환경 — 학습 곡선 · 거동", "결과");
  const cmds = "python train.py --env ev-threadpool --algo dqn --episodes 200\npython train.py --env ev-threadpool --algo ppo --episodes 300\npython evaluate.py --env ev-threadpool";
  [[0.55, "ev-threadpool 학습 곡선", "results/ev-threadpool_learning_curves.png"],
   [5.15, "ev-threadpool 정책 거동", "results/ev-threadpool_behavior.png"]].forEach(([x, t, p]) => {
    s.addShape(pres.shapes.RECTANGLE, { x, y: 1.05, w: 4.3, h: 2.6,
      fill: { color: "FAFAFA" }, line: { color: ORANGE, width: 1, dashType: "dash" } });
    s.addText("[ " + t + " ]", { x, y: 1.9, w: 4.3, h: 0.4, fontSize: 13, bold: true,
      color: ORANGE, fontFace: HFONT, align: "center", margin: 0 });
    s.addText(p + "  를 여기에 삽입", { x, y: 2.35, w: 4.3, h: 0.35, fontSize: 10,
      color: GRAY, fontFace: BFONT, align: "center", margin: 0 });
  });
  s.addText("실행 명령:", { x: 0.55, y: 3.95, w: 8.9, h: 0.3, fontSize: 11, bold: true,
    color: NAVY, fontFace: HFONT, margin: 0 });
  codeBox(s, cmds, { x: 0.55, y: 4.3, w: 8.9, h: 0.95, fs: 10 });
}

{
  const s = content("결과 ⑦ v2 — 정책 종합 비교 (p95 SLA 기준)", "결과");
  const rows = [
    [
      { text: "정책", options: { bold: true, color: "FFFFFF", fill: { color: NAVY }, align: "center" } },
      { text: "보상", options: { bold: true, color: "FFFFFF", fill: { color: NAVY }, align: "center" } },
      { text: "p95 SLA 위반률", options: { bold: true, color: "FFFFFF", fill: { color: NAVY }, align: "center" } },
      { text: "자원 비용", options: { bold: true, color: "FFFFFF", fill: { color: NAVY }, align: "center" } },
    ],
    [{ text: "DQN", options: { bold: true } }, "(기입)", "(기입)", "(기입)"],
    [{ text: "PPO", options: { bold: true } }, "(기입)", "(기입)", "(기입)"],
    [{ text: "Rule-based", options: { bold: true } }, "(기입)", "(기입)", "(기입)"],
    [{ text: "Static", options: { color: GRAY } }, "(기입)", "(기입)", "(기입)"],
  ];
  s.addText("ev-threadpool — evaluate.py 출력(results/ev-threadpool_summary.csv)을 그대로 옮겨 적기",
    { x: 0.55, y: 0.92, w: 8.9, h: 0.4, fontSize: 12, color: GRAY, fontFace: BFONT, margin: 0 });
  s.addTable(rows, { x: 0.55, y: 1.45, w: 5.4, colW: [1.4, 1.3, 1.5, 1.2], fontSize: 11.5,
    fontFace: BFONT, color: INK, border: { pt: 0.75, color: "CCCCCC" },
    align: "center", valign: "middle", rowH: 0.42 });
  s.addShape(pres.shapes.RECTANGLE, { x: 6.2, y: 1.45, w: 3.45, h: 2.1,
    fill: { color: "FAFAFA" }, line: { color: ORANGE, width: 1, dashType: "dash" } });
  s.addText("[ comparison 그래프 ]", { x: 6.2, y: 2.2, w: 3.45, h: 0.4, fontSize: 12,
    bold: true, color: ORANGE, fontFace: HFONT, align: "center", margin: 0 });
  bullets(s, [
    { t: "체크 포인트: v1(평균 SLA) 대비 v2(p95 SLA)에서 정책 순위가 유지되는가?", b: true },
    { t: "heavy-tail 환경에서는 '여유를 두는' 정책이 유리해질 수 있음 — 보상 가중치 해석과 함께 분석", lv: 1 },
  ], { x: 0.55, y: 4.1, w: 8.9, h: 1.1, fontSize: 12 });
}

// ================================================== 22. 결과 분석 종합
{
  const s = content("결과 분석 — 세 가지 분석 축", "결과");
  const findings = [
    ["1", "동적 제어는 필수인가 — Static과의 격차",
      "고정 thread(Static)는 버스트마다 SLA가 붕괴하는가? 동적 정책(룰·RL)과의 격차가 '튜닝 한 번 하고 방치'하는 관행의 비용을 정량화한다. (실행 결과로 채움)"],
    ["2", "보상 설계 = 운영 정책 설계 — c_base의 효과",
      "기본값 초과분에만 비용을 매기는 설계가 실제로 '평소 기본값 유지 → 피크 증설 → 복귀' 거동을 만들어내는가? 거동 그래프의 thread 궤적으로 확인한다."],
    ["3", "알고리즘 특성 — on-policy vs off-policy",
      "비단조 용량 + 드물지만 치명적인 backlog 상태에서 DQN(replay로 위기 경험 보존)과 PPO(현재 정책 데이터만 학습)의 강건성이 어떻게 갈리는가? v1·v2 공통으로 비교한다."],
  ];
  findings.forEach(([no, t, d], i) => {
    const y = 1.0 + i * 1.42;
    s.addShape(pres.shapes.RECTANGLE, { x: 0.55, y, w: 8.9, h: 1.26, fill: { color: LIGHT } });
    s.addShape(pres.shapes.RECTANGLE, { x: 0.55, y, w: 0.07, h: 1.26, fill: { color: ORANGE } });
    s.addText(no, { x: 0.75, y: y + 0.18, w: 0.6, h: 0.9, fontSize: 30, bold: true,
      color: ICE, fontFace: HFONT, margin: 0 });
    s.addText(t, { x: 1.45, y: y + 0.1, w: 7.85, h: 0.35, fontSize: 13.5, bold: true,
      color: NAVY, fontFace: HFONT, margin: 0 });
    s.addText(d, { x: 1.45, y: y + 0.48, w: 7.85, h: 0.72, fontSize: 11, color: INK,
      fontFace: BFONT, margin: 0 });
  });
}


// ================================================== 23. 한계
{
  const s = content("한계점 및 토론", "토론");
  bullets(s, [
    { t: "시뮬레이터와 실제의 간극 (sim-to-real gap)", b: true },
    { t: "시뮬레이터는 실제 JVM의 GC pause, JIT 워밍업, 요청 간 의존성을 단순화", lv: 1 },
    { t: "실제 적용 시 시뮬레이터 보정(실측 트래픽·지연 분포 반영)이 선행돼야 함", lv: 1 },
    { t: "PPO의 위기 상태 학습 한계", b: true },
    { t: "본 실험에서는 미해결 — 위기 상태 오버샘플링, 보상 정형화(clipping 완화), 커리큘럼 학습 등이 후보", lv: 1 },
    { t: "행동 공간의 단순화", b: true },
    { t: "±4 thread의 이산 행동 — 연속 제어(SAC)나 가변 폭 조절은 미실험", lv: 1 },
    { t: "단일 서비스 가정", b: true },
    { t: "실제 MSA는 서비스 간 의존(한 서비스의 병목이 전파) — 멀티에이전트 확장 필요", lv: 1 },
  ], { x: 0.55, y: 1.0, w: 8.9, h: 4.2, fontSize: 12.5 });
}

// ================================================== 24. 기여 & 향후
{
  const s = content("기여도 및 향후 연구", "결론");
  s.addText("기여", { x: 0.55, y: 0.95, w: 4.3, h: 0.35, fontSize: 15, bold: true,
    color: NAVY, fontFace: HFONT, margin: 0 });
  bullets(s, [
    { t: "운영 환경(고정 인스턴스 + Undertow)을 반영한 thread pool 튜닝 RL 환경을 충실도 2단계(v1 공식/v2 이산사건·p95)로 직접 설계·구현", fs: 11.5 },
    { t: "DQN·PPO를 교재 DeZero로 밑바닥 구현 — 알고리즘 동작을 코드 수준에서 검증", fs: 11.5 },
    { t: "실무 표준(임계값 룰)·고정 설정 대비 정량 비교 체계 구축 (시드 분리, 검증 best 체크포인트)", fs: 11.5 },
    { t: "c_base 기준 보상 설계로 '기본값 유지 + 피크 증설' 운영 정책을 보상 함수로 표현", fs: 11.5 },
  ], { x: 0.55, y: 1.35, w: 4.35, h: 3.2 });
  s.addText("향후 연구", { x: 5.25, y: 0.95, w: 4.3, h: 0.35, fontSize: 15, bold: true,
    color: ORANGE, fontFace: HFONT, margin: 0 });
  bullets(s, [
    { t: "실제 Undertow 기반 Spring Boot 서비스 연동 — 런타임 thread pool 조정 endpoint + JMeter 부하로 정책 검증", fs: 11.5, b: true },
    { t: "향후 확장: 수평 확장이 가능한 환경을 위한 인스턴스 오토스케일링 — 부록 환경으로 동일 프레임워크에서 실험 가능", fs: 11.5 },
    { t: "캡스톤 연구(MSA 로그 이상 탐지)와 통합: 이상 탐지가 찾은 위험 신호를 state에 추가해 장애 선제 대응형 스케일링으로 확장", fs: 11.5, b: true },
    { t: "실측 트래픽 로그 기반 시뮬레이터 보정 (sim-to-real)", fs: 11.5 },
  ], { x: 5.25, y: 1.35, w: 4.35, h: 3.4 });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.55, y: 4.58, w: 9.05, h: 0.02, fill: { color: ICE } });
  s.addText("제출물: PPT + 프로그램 소스 (autoscaling-rl/, 실행 방법 README 포함)",
    { x: 0.55, y: 4.7, w: 8.9, h: 0.35, fontSize: 11.5, color: GRAY, fontFace: BFONT, margin: 0 });
}

// ================================================== 25. 마무리
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addText("감사합니다", { x: 0.7, y: 2.05, w: 8.6, h: 0.9, fontSize: 40, bold: true,
    color: "FFFFFF", fontFace: HFONT, align: "center", margin: 0 });
  s.addText("Q & A", { x: 0.7, y: 3.05, w: 8.6, h: 0.5, fontSize: 18, color: ICE,
    fontFace: HFONT, align: "center", margin: 0 });
  s.addShape(pres.shapes.RECTANGLE, { x: 4.6, y: 2.0, w: 0.8, h: 0.035, fill: { color: ORANGE } });
}

const OUT = process.env.OUT || "autoscaling_rl_발표.pptx";  // OUT=이름.pptx node build_deck.js
pres.writeFile({ fileName: path.join(__dirname, OUT) })
  .then(() => console.log("PPT 생성 완료"));
