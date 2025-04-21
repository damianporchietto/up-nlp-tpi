const puppeteer = require('puppeteer')
const fs = require('fs')

const scrapData = async (page, ministry) => {
    console.log('Scraping data')
    const url = await page.url()
    const hash = url.split('/').pop()

    await page.waitForSelector('.contenido-title')
    await page.waitForSelector('.descripcion')
    await page.waitForSelector('mat-accordion')

    const title = await page.evaluate(() => {
        const title = document.querySelector('.contenido-title')
        return window.textClear(title?.innerText || '')
    })

    const description = await page.evaluate(() => {
        const description = document.querySelector('.descripcion')
        return window.textClear(description?.innerText || '')
    })

    const requirements = await page.evaluate(() => {
        const requirements =  document.querySelector('mat-accordion').querySelectorAll('mat-expansion-panel')
        const requirementsArray = []
        
        for(const requirement of requirements) {
            const title = requirement.querySelector('mat-expansion-panel-header').innerText
            const content = requirement.querySelector('.mat-expansion-panel-content').innerText
            requirementsArray.push({
                title: window.textClear(title),
                content: window.textClear(content)
            })
        }

        return requirementsArray
    })

    const data = {
        url,
        title,
        description,
        requirements
    }

    console.log(data)

    fs.mkdirSync(`${ministry}`, { recursive: true })
    fs.writeFileSync(`${ministry}/${hash}.json`, JSON.stringify(data, null, 2))
}

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms))

const main = async () => {
    const browser = await puppeteer.launch({ headless: false })
    const page = await browser.newPage()

    await page.on('framenavigated', async () => {
        await page.evaluate(() => {
            window.textClear = (text) => {
                return text
                .replace(/[\u200B-\u200D\uFEFF\u2028\u2029]/g, '') 
                .replace(/\s+/g, ' ')
                .replace(/[\r\n\t]+/g, ' ')
                .trim()
            }
        })
    })

    await page.goto('https://cidi.cba.gov.ar/portal-publico/resultado-busqueda')
    
    const select = 'body > app-root > app-sidebar > mat-drawer-container > mat-drawer-content > app-resultado-busqueda > div > div.filtro-serv.ng-star-inserted > div > div > app-filtro > div > mat-form-field'

    await page.waitForSelector(select)    
    await page.click(select)    

    const globalMinistryIndex = await page.evaluate(() => {
        const options = document
            .querySelectorAll("mat-option")
        return options.length
    })

    console.log({globalMinistryIndex})

    for(let i = 0; i < globalMinistryIndex; i++) {
        await page.waitForSelector(select)    
        await page.click(select)

        const ministry = await page.evaluate((index) => {
            const options = document
                .querySelectorAll("mat-option")
            options[index].click()
            return options[index].innerText
        }, i)

        await sleep(5000)

        const throughIndex = await page.evaluate(() => {
            const options = document
                .querySelector('.wrapper-guia-tramites .card-container')
                ?.querySelectorAll('mat-card')
            return options?.length || 0
        })

        console.log({throughIndex})

        if(throughIndex === 0) {
            console.log('No hay tramites')
            continue
        }

        for(let i = 0; i < throughIndex; i++) {
            await page.evaluate((index) => {
                const options = document
                .querySelector('.wrapper-guia-tramites .card-container')
                .querySelectorAll('mat-card')

                options[index].click()
            }, i)

            await scrapData(page, ministry)
            await sleep(1000)
            await page.goBack()
            await sleep(1000)
        }
    }
    
    await browser.close()
}

if(require.main === module) {
    main().then(() => {
        console.log('Done')
    }).catch((err) => {
        console.error(err)
    })
}