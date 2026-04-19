<script lang="ts">
	import { authStore } from '$lib/stores/auth.svelte';
	import { goto } from '$app/navigation';

	let fullName = $state(authStore.user?.full_name || '');
	let email = $state(authStore.user?.email || '');
	let profileSaved = $state(false);
	let profileError = $state('');

	let oldPassword = $state('');
	let newPassword = $state('');
	let confirmPassword = $state('');
	let passwordSaved = $state(false);
	let passwordError = $state('');

	let passwordMismatch = $derived(confirmPassword.length > 0 && newPassword !== confirmPassword);

	let emailRecipients = $state<string[]>(authStore.user?.email_recipients || []);
	let newRecipient = $state('');
	let defaultRecipient = $state(authStore.user?.default_recipient || '');
	let emailSaved = $state(false);
	let emailError = $state('');

	$effect(() => {
		if (authStore.user) {
			fullName = authStore.user.full_name;
			email = authStore.user.email;
			emailRecipients = authStore.user.email_recipients;
			defaultRecipient = authStore.user.default_recipient;
		}
	});

	$effect(() => {
		if (!authStore.isAuthenticated && !authStore.loading) {
			goto('/auth/login');
		}
	});

	async function handleProfileSave() {
		profileSaved = false;
		profileError = '';
		const success = await authStore.updateProfile(fullName || null, email || null);
		if (success) {
			profileSaved = true;
			setTimeout(() => { profileSaved = false; }, 3000);
		} else {
			profileError = 'Ошибка сохранения профиля';
		}
	}

	async function handlePasswordChange() {
		if (passwordMismatch) return;
		passwordSaved = false;
		passwordError = '';
		const error = await authStore.changePassword(oldPassword, newPassword);
		if (error) {
			passwordError = error;
		} else {
			passwordSaved = true;
			oldPassword = '';
			newPassword = '';
			confirmPassword = '';
			setTimeout(() => { passwordSaved = false; }, 3000);
		}
	}

	function addRecipient() {
		const trimmed = newRecipient.trim();
		if (trimmed && !emailRecipients.includes(trimmed)) {
			emailRecipients = [...emailRecipients, trimmed];
			if (!defaultRecipient) defaultRecipient = trimmed;
			newRecipient = '';
		}
	}

	function removeRecipient(addr: string) {
		emailRecipients = emailRecipients.filter(r => r !== addr);
		if (defaultRecipient === addr) {
			defaultRecipient = emailRecipients[0] || '';
		}
	}

	async function handleEmailSave() {
		emailSaved = false;
		emailError = '';
		const success = await authStore.updateEmailSettings(emailRecipients, defaultRecipient);
		if (success) {
			emailSaved = true;
			setTimeout(() => { emailSaved = false; }, 3000);
		} else {
			emailError = 'Ошибка сохранения настроек';
		}
	}

	let initials = $derived(
		authStore.user?.full_name
			? authStore.user.full_name.split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase()
			: authStore.user?.username?.slice(0, 2).toUpperCase() || '?'
	);
</script>

<div class="max-w-4xl mx-auto space-y-6">
	<div class="flex items-center justify-between animate-fade-in">
		<div>
			<h2 class="text-2xl font-bold text-text">Профиль</h2>
			<p class="text-sm text-text-secondary mt-1">Настройки аккаунта и параметры отправки</p>
		</div>
	</div>

	<div class="bg-elevated border border-border rounded-xl p-6 animate-fade-in">
		<div class="flex items-center gap-4 mb-6">
			<div class="h-14 w-14 rounded-xl bg-accent/10 border border-accent/20 flex items-center justify-center">
				<span class="text-lg font-bold text-accent">{initials}</span>
			</div>
			<div>
				<h3 class="text-lg font-semibold text-text">{authStore.user?.full_name || authStore.user?.username}</h3>
				<p class="text-sm text-text-tertiary font-mono">{authStore.user?.username} / {authStore.user?.email}</p>
			</div>
		</div>

		<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
			<div>
				<label for="fullName" class="block text-xs font-medium text-text-secondary mb-1.5">ФИО</label>
				<input id="fullName" type="text" bind:value={fullName}
					class="w-full px-3 py-2.5 bg-base border border-border rounded-lg text-text text-sm
						focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/30 transition-colors duration-200" />
			</div>
			<div>
				<label for="email" class="block text-xs font-medium text-text-secondary mb-1.5">Email</label>
				<input id="email" type="email" bind:value={email}
					class="w-full px-3 py-2.5 bg-base border border-border rounded-lg text-text font-mono text-sm
						focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/30 transition-colors duration-200" />
			</div>
		</div>

		<div class="mt-4 flex items-center gap-3">
			<button onclick={handleProfileSave}
				class="px-5 py-2 bg-accent text-white rounded-lg font-medium text-sm transition-all duration-200 hover:bg-accent-hover active:scale-[0.98]">
				Сохранить профиль
			</button>
			{#if profileSaved}<span class="text-sm text-success animate-fade-in">Сохранено!</span>{/if}
			{#if profileError}<span class="text-sm text-danger animate-fade-in">{profileError}</span>{/if}
		</div>
	</div>

	<div id="password" class="bg-elevated border border-border rounded-xl p-6 animate-fade-in">
		<div class="flex items-center gap-3 mb-6">
			<div class="h-9 w-9 rounded-lg bg-warning/10 border border-warning/20 flex items-center justify-center">
				<svg class="w-4 h-4 text-warning" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
					<path stroke-linecap="round" stroke-linejoin="round" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
				</svg>
			</div>
			<h3 class="text-lg font-semibold text-text">Смена пароля</h3>
		</div>

		<div class="grid grid-cols-1 md:grid-cols-3 gap-4">
			<div>
				<label for="oldPassword" class="block text-xs font-medium text-text-secondary mb-1.5">Текущий пароль</label>
				<input id="oldPassword" type="password" bind:value={oldPassword}
					class="w-full px-3 py-2.5 bg-base border border-border rounded-lg text-text font-mono text-sm
						focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/30 transition-colors duration-200" />
			</div>
			<div>
				<label for="newPassword" class="block text-xs font-medium text-text-secondary mb-1.5">Новый пароль</label>
				<input id="newPassword" type="password" bind:value={newPassword}
					class="w-full px-3 py-2.5 bg-base border border-border rounded-lg text-text font-mono text-sm
						focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/30 transition-colors duration-200" />
			</div>
			<div>
				<label for="confirmPassword" class="block text-xs font-medium text-text-secondary mb-1.5">Подтверждение</label>
				<input id="confirmPassword" type="password" bind:value={confirmPassword}
					class="w-full px-3 py-2.5 bg-base border border-border rounded-lg text-text font-mono text-sm
						focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/30 transition-colors duration-200
						{passwordMismatch ? 'border-danger' : ''}" />
				{#if passwordMismatch}<p class="text-xs text-danger mt-1">Пароли не совпадают</p>{/if}
			</div>
		</div>

		<div class="mt-4 flex items-center gap-3">
			<button onclick={handlePasswordChange} disabled={!oldPassword || !newPassword || passwordMismatch}
				class="px-5 py-2 bg-warning text-base rounded-lg font-medium text-sm transition-all duration-200 hover:bg-warning/90 active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed">
				Сменить пароль
			</button>
			{#if passwordSaved}<span class="text-sm text-success animate-fade-in">Пароль изменён!</span>{/if}
			{#if passwordError}<span class="text-sm text-danger animate-fade-in">{passwordError}</span>{/if}
		</div>
	</div>

	<div id="email" class="bg-elevated border border-border rounded-xl p-6 animate-fade-in">
		<div class="flex items-center gap-3 mb-6">
			<div class="h-9 w-9 rounded-lg bg-accent/10 border border-accent/20 flex items-center justify-center">
				<svg class="w-4 h-4 text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
					<path stroke-linecap="round" stroke-linejoin="round" d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
				</svg>
			</div>
			<h3 class="text-lg font-semibold text-text">Настройки Email для отправки XML</h3>
		</div>

		<div class="mb-4">
			<label id="recipients-label" class="block text-xs font-medium text-text-secondary mb-2">Адреса получателей</label>
			<div class="space-y-2">
				{#each emailRecipients as addr}
					<div class="flex items-center justify-between px-3 py-2 bg-base border border-border rounded-lg">
						<div class="flex items-center gap-3">
							<button
								onclick={() => defaultRecipient = addr}
								class="h-4 w-4 rounded-full border-2 transition-colors duration-200
									{defaultRecipient === addr ? 'bg-accent border-accent' : 'border-text-text-tertiary hover:border-accent/50'}"
								title="Использовать по умолчанию"
							></button>
							<span class="text-sm text-text font-mono">{addr}</span>
							{#if defaultRecipient === addr}
								<span class="text-[10px] px-1.5 py-0.5 bg-accent/10 text-accent rounded font-mono">по умолчанию</span>
							{/if}
						</div>
						<button onclick={() => removeRecipient(addr)} class="text-text-tertiary hover:text-danger transition-colors duration-200" title="Удалить">
							<svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
								<path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" />
							</svg>
						</button>
					</div>
				{/each}
			</div>
		</div>

		<div class="flex gap-3">
			<input type="email" bind:value={newRecipient} placeholder="email@example.com"
				onkeydown={(e) => e.key === 'Enter' && addRecipient()}
				class="flex-1 px-3 py-2.5 bg-base border border-border rounded-lg text-text font-mono text-sm
					placeholder:text-text-tertiary/50
					focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/30 transition-colors duration-200" />
			<button onclick={addRecipient} disabled={!newRecipient.trim()}
				class="px-4 py-2.5 bg-overlay border border-border text-accent rounded-lg text-sm font-medium
					transition-all duration-200 hover:bg-overlay active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed">
				Добавить
			</button>
		</div>

		<div class="mt-4 flex items-center gap-3">
			<button onclick={handleEmailSave}
				class="px-5 py-2 bg-accent text-white rounded-lg font-medium text-sm transition-all duration-200 hover:bg-accent-hover active:scale-[0.98]">
				Сохранить настройки
			</button>
			{#if emailSaved}<span class="text-sm text-success animate-fade-in">Сохранено!</span>{/if}
			{#if emailError}<span class="text-sm text-danger animate-fade-in">{emailError}</span>{/if}
		</div>
	</div>
</div>
