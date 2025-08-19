const chat = document.getElementById('chat');
const input = document.getElementById('msg');
const sendBtn = document.getElementById('send');

function addUserMessage(text){
  const row = document.createElement('div');
  row.className = 'row user';
  row.innerHTML = `
    <div class="avatar user">🧑</div>
    <div class="bubble user">
      <div class="content"></div>
    </div>
  `;
  row.querySelector('.content').textContent = text;
  chat.appendChild(row);
  chat.scrollTop = chat.scrollHeight;
}

function addBotMessage(text, phaseClass){
  const row = document.createElement('div');
  row.className = 'row';
  row.innerHTML = `
    <div class="avatar">🤖</div>
    <div class="bubble ${phaseClass || 'eligibility'}">
      <div class="meta"><span class="badge ${phaseClass || 'eligibility'}">${labelFor(phaseClass)}</span></div>
      <div class="content"></div>
    </div>
  `;
  row.querySelector('.content').textContent = text;
  chat.appendChild(row);
  chat.scrollTop = chat.scrollHeight;
}

function labelFor(phase){
  switch(phase){
    case 'eligibility': return 'Eligibility';
    case 'listings': return 'Listings';
    case 'area': return 'Area';
    case 'done': return 'Done';
    default: return 'Assistant';
  }
}

async function send(){
  const text = input.value.trim();
  if(!text) return;
  addUserMessage(text);
  input.value = '';
  try{
    const res = await fetch('/api/message', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text })
    });
    const data = await res.json();
    if(!res.ok){
      addBotMessage(data.error || 'Error', 'eligibility');
      return;
    }
    addBotMessage(data.reply, data.phaseClass);
  }catch(err){
    addBotMessage('Network error', 'eligibility');
  }
}

sendBtn.addEventListener('click', send);
input.addEventListener('keydown', (e)=>{ if(e.key === 'Enter') send(); });

document.querySelectorAll('.chip').forEach(btn=>{
  btn.addEventListener('click', ()=>{ input.value = btn.dataset.text; input.focus(); });
});

// Greeting
addBotMessage('Hi! Tell me about your income, debts, savings and budget (e.g., "I earn 110k, debts 450/mo, savings 40k; budget 800k in Brunswick"). Then use /next to see listings.', 'eligibility');