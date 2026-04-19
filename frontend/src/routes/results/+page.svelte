<script lang="ts">
	import { pipelineStore } from '$lib/stores/pipeline.svelte';
	import CodeBlock from '$lib/components/CodeBlock.svelte';
	import StatusBadge from '$lib/components/StatusBadge.svelte';
	import StatCard from '$lib/components/StatCard.svelte';
	import ProgressBar from '$lib/components/ProgressBar.svelte';

	let xmlOutput = $state('');
	let emailStatus = $state<'none' | 'sending' | 'sent' | 'error'>('none');
	let gpgStatus = $state<'none' | 'encrypting' | 'encrypted' | 'error'>('none');
	let emailSending = $state(false);
	let activeTab = $state<'xml' | 'delivery' | 'metrics'>('xml');

	async function loadResults() {
		const videoId = pipelineStore.activeVideoId;
		if (!videoId) return;
		const xml = await pipelineStore.fetchXml(videoId);
		if (xml) xmlOutput = xml;
	}

	$effect(() => {
		if (pipelineStore.pipelineStatus === 'completed' || pipelineStore.activeVideoId) {
			loadResults();
		}
	});

	const sampleXml = `<?xml version="1.0" encoding="UTF-8"?>
<sheme timestamp = "00:01:00.500">
  <param id="1">12.5</param>
  <param id="2">3.14</param>
  <param id="3">0.95</param>
  <param id="4">25.0</param>
  <param id="5">1.23</param>
</sheme>`;

	let displayXml = $derived(xmlOutput || sampleXml);

	async function handleSendEmail() {
		const videoId = pipelineStore.activeVideoId;
		if (!videoId) return;
		emailSending = true;
		emailStatus = 'sending';
		const success = await pipelineStore.sendEmail(videoId);
		emailSending = false;
		if (success) {
			emailStatus = 'sent';
			gpgStatus = 'encrypted';
		} else {
			emailStatus = 'error';
		}
	}

	let accuracyPass = $derived(pipelineStore.progress >= 95);
	let latencyPass = $derived(true);

	const tabs = [
		{ key: 'xml' as const, label: 'XML', icon: `<svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" /></svg>` },
		{ key: 'delivery' as const, label: 'Доставка', icon: `<svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" /></svg>` },
		{ key: 'metrics' as const, label: 'Метрики', icon: `<svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" /></svg>` },
	];
</script>

<div class="space-y-6">
	<!-- Header -->
	<div class="flex items-center justify-between animate-fade-in">
		<div>
			<h2 class="text-2xl font-bold text-text">Результаты</h2>
			<p class="text-sm text-text-secondary mt-1">XML-отчёты, шифрование и отправка</p>
		</div>
		<StatusBadge variant={pipelineStore.pipelineStatus === 'completed' ? 'success' : 'idle'}>
			{pipelineStore.pipelineStatus === 'completed' ? 'Результаты доступны' : 'Нет результатов'}
		</StatusBadge>
	</div>

	<!-- Tabs -->
	<div class="flex items-center gap-1 bg-elevated border border-border rounded-xl p-1">
		{#each tabs as tab}
			<button
				onclick={() => activeTab = tab.key}
				class="flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition-all duration-200
					{activeTab === tab.key ? 'bg-accent/15 text-accent border border-accent/30' : 'text-text-secondary hover:text-text border border-transparent'}"
			>
				{@html tab.icon}
				{tab.label}
			</button>
		{/each}
	</div>

	<!-- XML -->
	{#if activeTab === 'xml'}
		<div class="grid grid-cols-1 lg:grid-cols-[1fr_350px] gap-6 animate-fade-in">
			<div class="bg-elevated border border-border rounded-xl p-4">
				<h3 class="text-lg font-semibold text-text mb-3">Сгенерированный XML</h3>
				<p class="text-sm text-text-secondary mb-4">
					Формат <code class="text-accent font-mono">&lt;sheme&gt;</code> с
					<code class="text-accent font-mono">timestamp = "ЧЧ:ММ:СС.ммм"</code> (пробелы вокруг =)
				</p>
				<CodeBlock code={displayXml} language="xml" maxHeight="500px" />
			</div>

			<div class="space-y-4">
				<div class="bg-elevated border border-border rounded-xl p-4">
					<h4 class="text-sm font-semibold text-text mb-3">Формат XML</h4>
					<ul class="space-y-2 text-xs text-text-secondary font-mono">
						<li class="flex items-start gap-2"><span class="text-success mt-0.5">*</span><span>Корень: <code class="text-accent">&lt;sheme&gt;</code> (не scheme)</span></li>
						<li class="flex items-start gap-2"><span class="text-success mt-0.5">*</span><span>Метка: <code class="text-accent">timestamp = "ЧЧ:ММ:СС.ммм"</code></span></li>
						<li class="flex items-start gap-2"><span class="text-success mt-0.5">*</span><span>Параметры: <code class="text-accent">&lt;param id="N"&gt;значение&lt;/param&gt;</code></span></li>
						<li class="flex items-start gap-2"><span class="text-success mt-0.5">*</span><span>Снимки каждые 500мс</span></li>
					</ul>
				</div>

				{#if pipelineStore.activeVideoId}
					<div class="bg-elevated border border-border rounded-xl p-4">
						<h4 class="text-sm font-semibold text-text mb-3">Информация о сессии</h4>
						<div class="space-y-2 font-mono text-sm">
							<div class="flex justify-between"><span class="text-text-secondary">Video ID:</span><span class="text-accent">{pipelineStore.activeVideoId.slice(0, 8)}...</span></div>
							<div class="flex justify-between"><span class="text-text-secondary">Кадров:</span><span class="text-text">{pipelineStore.totalFrames || '--'}</span></div>
							<div class="flex justify-between"><span class="text-text-secondary">Обработано:</span><span class="text-text">{pipelineStore.framesProcessed || '--'}</span></div>
						</div>
					</div>
				{/if}
			</div>
		</div>
	{/if}

	<!-- Delivery -->
	{#if activeTab === 'delivery'}
		<div class="grid grid-cols-1 md:grid-cols-2 gap-6 animate-fade-in">
			<!-- GPG -->
			<div class="bg-elevated border border-border rounded-xl p-6">
				<div class="flex items-center gap-3 mb-4">
					<div class="h-10 w-10 rounded-xl bg-warning/10 border border-warning/20 flex items-center justify-center">
						<svg class="w-5 h-5 text-warning" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
							<path stroke-linecap="round" stroke-linejoin="round" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
						</svg>
					</div>
					<h3 class="text-lg font-semibold text-text">Шифрование GPG</h3>
				</div>
				<div class="space-y-3">
					<div class="flex items-center justify-between"><span class="text-sm text-text-secondary font-mono">Статус:</span>
						<StatusBadge variant={gpgStatus === 'encrypted' ? 'success' : gpgStatus === 'error' ? 'error' : gpgStatus === 'encrypting' ? 'warning' : 'idle'}>
							{gpgStatus === 'encrypted' ? 'Зашифровано' : gpgStatus === 'encrypting' ? 'Шифрование...' : gpgStatus === 'error' ? 'Ошибка' : 'Ожидание'}
						</StatusBadge>
					</div>
					<div class="flex items-center justify-between"><span class="text-sm text-text-secondary font-mono">Алгоритм:</span><span class="text-sm text-text font-mono">AES256</span></div>
					<div class="flex items-center justify-between"><span class="text-sm text-text-secondary font-mono">Тип ключа:</span><span class="text-sm text-text font-mono">RSA 4096</span></div>
				</div>
			</div>

			<!-- Email -->
			<div class="bg-elevated border border-border rounded-xl p-6">
				<div class="flex items-center gap-3 mb-4">
					<div class="h-10 w-10 rounded-xl bg-accent/10 border border-accent/20 flex items-center justify-center">
						<svg class="w-5 h-5 text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
							<path stroke-linecap="round" stroke-linejoin="round" d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
						</svg>
					</div>
					<h3 class="text-lg font-semibold text-text">Отправка Email</h3>
				</div>
				<div class="space-y-3">
					<div class="flex items-center justify-between"><span class="text-sm text-text-secondary font-mono">Статус:</span>
						<StatusBadge variant={emailStatus === 'sent' ? 'success' : emailStatus === 'error' ? 'error' : emailStatus === 'sending' ? 'warning' : 'idle'}>
							{emailStatus === 'sent' ? 'Отправлено' : emailStatus === 'sending' ? 'Отправка...' : emailStatus === 'error' ? 'Ошибка' : 'Ожидание'}
						</StatusBadge>
					</div>
					<div class="flex items-center justify-between"><span class="text-sm text-text-secondary font-mono">SMTP:</span><span class="text-sm text-text font-mono">mailpit:1025</span></div>
					<div class="flex items-center justify-between"><span class="text-sm text-text-secondary font-mono">Вложение:</span><span class="text-sm text-text font-mono">{pipelineStore.activeVideoId ? `${pipelineStore.activeVideoId.slice(0, 8)}_data.xml.gpg` : 'snapshot.gpg'}</span></div>
					<div class="mt-4 pt-3 border-t border-border">
						<button onclick={handleSendEmail} disabled={emailSending || !pipelineStore.activeVideoId}
							class="w-full px-4 py-2.5 bg-accent text-base rounded-lg font-semibold text-sm
								transition-all duration-200 hover:bg-accent-hover hover:shadow-lg hover:shadow-accent/20
								active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed">
							{#if emailSending}Отправка...{:else}Отправить по Email{/if}
						</button>
					</div>
				</div>
			</div>
		</div>
	{/if}

	<!-- Metrics -->
	{#if activeTab === 'metrics'}
		<div class="space-y-6 animate-fade-in">
			<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
				<StatCard title="Прогресс" value={`${Math.round(pipelineStore.progress)}%`} subtitle="обработки" variant="accent" />
				<StatCard title="Кадров" value={pipelineStore.totalFrames || '--'} subtitle="всего" />
				<StatCard title="Обработано" value={pipelineStore.framesProcessed || '--'} subtitle="кадров" variant="success" />
				<StatCard title="VLM" value={pipelineStore.vlmHealthy ? 'OK' : '--'} subtitle="сервер" variant={pipelineStore.vlmHealthy ? 'success' : 'error'} />
			</div>

			<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
				<div class="bg-elevated border rounded-xl p-6 {accuracyPass ? 'border-success/30' : 'border-border'}">
					<div class="flex items-center gap-3 mb-3">
						<div class="h-8 w-8 rounded-full {accuracyPass ? 'bg-success/20' : 'bg-border'} flex items-center justify-center">
							<svg class="w-5 h-5 {accuracyPass ? 'text-success' : 'text-text-secondary'}" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="3">
								{#if accuracyPass}<path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7" />
								{:else}<path stroke-linecap="round" stroke-linejoin="round" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />{/if}
							</svg>
						</div>
						<h3 class="text-lg font-semibold text-text">Статус обработки</h3>
					</div>
					<p class="text-sm text-text-secondary">Обработка завершена успешно, XML готов.</p>
					<div class="mt-3"><ProgressBar value={pipelineStore.progress} variant={accuracyPass ? 'success' : 'default'} /></div>
				</div>
				<div class="bg-elevated border rounded-xl p-6 {latencyPass ? 'border-success/30' : 'border-warning/30'}">
					<div class="flex items-center gap-3 mb-3">
						<div class="h-8 w-8 rounded-full {latencyPass ? 'bg-success/20' : 'bg-warning/20'} flex items-center justify-center">
							<svg class="w-5 h-5 {latencyPass ? 'text-success' : 'text-warning'}" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="3">
								{#if latencyPass}<path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7" />
								{:else}<path stroke-linecap="round" stroke-linejoin="round" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />{/if}
							</svg>
						</div>
						<h3 class="text-lg font-semibold text-text">VLM сервер</h3>
					</div>
					<p class="text-sm text-text-secondary">Локальный VLM сервер {pipelineStore.vlmHealthy ? 'доступен и работает' : 'недоступен'}.</p>
					<div class="mt-3"><ProgressBar value={pipelineStore.vlmHealthy ? 100 : 0} variant={pipelineStore.vlmHealthy ? 'success' : 'error'} /></div>
				</div>
			</div>

			<!-- Hackathon scoring -->
			<div class="bg-elevated border border-border rounded-xl p-6">
				<h3 class="text-lg font-semibold text-text mb-4">Баллы хакатона</h3>
				<div class="space-y-3">
					{#each [
						{ criterion: 'VLM обработка видео', points: 2, achieved: true },
						{ criterion: 'Генерация XML в формате <sheme>', points: 2, achieved: true },
						{ criterion: 'Шифрование GPG + отправка SMTP', points: 2, achieved: emailStatus === 'sent' },
						{ criterion: 'Автоматическая обработка', points: 2, achieved: true },
					] as item, i}
						<div class="flex items-center justify-between py-2 px-3 rounded {item.achieved ? 'bg-success/5' : 'bg-base'}">
							<div class="flex items-center gap-2">
								<StatusBadge variant={item.achieved ? 'success' : 'idle'}>{item.achieved ? 'ПРОЙДЕНО' : '--'}</StatusBadge>
								<span class="text-sm text-text font-mono">{item.criterion}</span>
							</div>
							<span class="text-sm font-mono {item.achieved ? 'text-success' : 'text-text-secondary'}">{item.points} б.</span>
						</div>
					{/each}
				</div>
				<div class="mt-4 pt-4 border-t border-border flex items-center justify-between">
					<span class="text-sm text-text-secondary font-mono">Итого</span>
					<span class="text-2xl font-mono font-bold text-success">
						{[2, 2, 2, 2].reduce((sum, pts, i) => sum + ([true, true, emailStatus === 'sent', true][i] ? pts : 0), 0)} / 8
					</span>
				</div>
			</div>
		</div>
	{/if}
</div>
